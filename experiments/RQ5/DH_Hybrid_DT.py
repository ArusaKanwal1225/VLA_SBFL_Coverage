"""
DeepHyperion-style Illumination Search (MAP-Elites) for pi0 VLA testing on SimplerEnv,
fused with 3D TRAJECTORY COVERAGE (spatial + directional) AND a DECISION-TREE FAILURE
SURROGATE for surrogate-assisted candidate selection (SAIL-style).
"""

import os
import json
import math
import random
import subprocess
from collections import defaultdict, deque
import argparse
import time

import torch
from tqdm import tqdm
from model_interface_allMetrics import VLAInterface
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from run_fuzzer_allMetrics_integrated import SimRunner

# USER CONFIG

NUM_RUNS = 1

X_RANGE = (-0.5, -0.05)
Y_RANGE = (0.0, 0.4)
Z_RANGE = (0.0, 0.5)

# --- TCP WORKSPACE voxel grid (the volume the GRIPPER sweeps through) ---
VOX_X = (-0.65, 0.05)
VOX_Y = (-0.15, 0.65)
VOX_Z = (0.80, 1.15)

VOXEL_SIZE = 0.05

#BASE_ROOT = "../data/DH_HybridDT_V1_500scenes"

#PYTHON_BIN = "/root/anaconda3/envs/EO1/bin/python"
#FUZZER = "run_fuzzer_allMetrics.py"

ORIENTATIONS = {
    1: [0.707, 0.707, 0.0, 0.0],
    2: [0.707, 0.0, 0.0, 0.707],
    3: [0.707, 0.0, 0.707, 0.0],
    4: [0.98, 0.0, 0.0, 0.20],
    5: [0.20, 0.0, 0.0, 0.98],
}

OBJECTS = {
    "pick_redbull_can": "opened_redbull_can",
    "pick_blue_plastic_bottle": "blue_plastic_bottle",
    "pick_eggplant": "eggplant",
    "pick_carrot": "bridge_carrot_generated_modified",
    "pick_apple": "apple",
    "pick_spoon": "bridge_spoon_generated_modified",
    "pick_sponge": "sponge",
}

# SEARCH CONFIG


SEED_POOL_SIZE = 20
POP_SIZE = 10
GRID_SIZE = 25

MAP_FEATURES = ("obj_x", "obj_y")
ALIGN_TRAJECTORY = False

# 3D TRAJECTORY-COVERAGE CONFIG  (spatial + directional)

N_AZ = 12
N_EL = 6

COVERAGE_LAMBDA = 0.05

P_FRONTIER = 0.3
FRONTIER_SIZE = 15

MP_THRESHOLD = 0.8
MP_CI_LOWER_THRESHOLD = 0.65

# SURROGATE CONFIG  (decision-tree failure predictor + SAIL acquisition)

USE_SURROGATE = True

SURROGATE_CANDIDATES = 12
SURROGATE_RETRAIN_EVERY = 5
SURROGATE_MAX_DEPTH = 8
SURROGATE_EPSILON = 0.15
W_FAIL = 1.0
W_NOVELTY = 0.6
W_DIVERSITY = 0.3

STRUCTURAL_FEATURES = {"obj_x", "obj_y", "reach_dist", "orient"}

# WORKSPACE VOXEL GRID  (set V) and DIRECTION CONES (set C)

NX = len(np.arange(VOX_X[0], VOX_X[1], VOXEL_SIZE))
NY = len(np.arange(VOX_Y[0], VOX_Y[1], VOXEL_SIZE))
NZ = len(np.arange(VOX_Z[0], VOX_Z[1], VOXEL_SIZE))
NUM_VOXELS = NX * NY * NZ
NUM_CONES = N_AZ * N_EL


def point_voxel(p):
    ix = int((p[0] - VOX_X[0]) / VOXEL_SIZE)
    iy = int((p[1] - VOX_Y[0]) / VOXEL_SIZE)
    iz = int((p[2] - VOX_Z[0]) / VOXEL_SIZE)
    if 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ:
        return (ix, iy, iz)
    return None


def segment_cone(a, b):
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        return None
    dx, dy, dz = dx / n, dy / n, dz / n
    az = math.atan2(dy, dx)
    el = math.asin(max(-1.0, min(1.0, dz)))
    az_bin = min(N_AZ - 1, max(0, int((az + math.pi) / (2 * math.pi) * N_AZ)))
    el_bin = min(N_EL - 1, max(0, int((el + math.pi / 2) / math.pi * N_EL)))
    return (az_bin, el_bin)


def trajectory_coverage(traj):
    vox, cones = set(), set()
    if not traj:
        return vox, cones
    v0 = point_voxel(traj[0])
    if v0:
        vox.add(v0)
    for i in range(1, len(traj)):
        a, b = traj[i - 1], traj[i]
        seg = math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3)))
        steps = max(1, int(seg / (VOXEL_SIZE * 0.5)))
        for s in range(1, steps + 1):
            t = s / steps
            p = (a[0] + (b[0] - a[0]) * t,
                 a[1] + (b[1] - a[1]) * t,
                 a[2] + (b[2] - a[2]) * t)
            v = point_voxel(p)
            if v:
                vox.add(v)
        c = segment_cone(a, b)
        if c:
            cones.add(c)
    return vox, cones


# SCENE  (genome)

class Scene:
    def __init__(self):
        self.x = random.uniform(*X_RANGE)
        self.y = random.uniform(*Y_RANGE)
        self.orientation_id = random.choice(list(ORIENTATIONS.keys()))
        self.object_name = random.choice(list(OBJECTS.keys()))

    def clone(self):
        s = Scene.__new__(Scene)
        s.x, s.y = self.x, self.y
        s.orientation_id = self.orientation_id
        s.object_name = self.object_name
        return s

    def mutate(self):
        new = self.clone()
        if random.random() < 0.2:
            new.x = random.uniform(*X_RANGE)
            new.y = random.uniform(*Y_RANGE)
        else:
            step = random.uniform(0.02, 0.1)
            new.x = float(np.clip(self.x + random.uniform(-step, step), *X_RANGE))
            new.y = float(np.clip(self.y + random.uniform(-step, step), *Y_RANGE))
        if random.random() < 0.5:
            new.orientation_id = random.choice(list(ORIENTATIONS.keys()))
        if random.random() < 0.5:
            new.object_name = random.choice(list(OBJECTS.keys()))
        return new

    def differs_from(self, other):
        return (abs(self.x - other.x) > 1e-4 or abs(self.y - other.y) > 1e-4
                or self.orientation_id != other.orientation_id
                or self.object_name != other.object_name)

# TRAJECTORY HELPERS

def align_trajectory(traj):
    if not traj:
        return traj
    xs, ys, zs = [p[0] for p in traj], [p[1] for p in traj], [p[2] for p in traj]
    cx, cy, cz = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2
    tx, ty, tz = sum(X_RANGE)/2, sum(Y_RANGE)/2, sum(Z_RANGE)/2
    dx, dy, dz = tx-cx, ty-cy, tz-cz
    return [(x+dx, y+dy, z+dz) for x, y, z in traj]


def path_length(traj):
    if len(traj) < 2:
        return 0.0
    return sum(math.sqrt(sum((traj[i+1][k]-traj[i][k])**2 for k in range(3)))
               for i in range(len(traj)-1))

# FEATURE LIBRARY  (map axes)

FEATURE_LIBRARY = {
    "obj_x":      (lambda s, t: s.x,                       X_RANGE[0], X_RANGE[1]),
    "obj_y":      (lambda s, t: s.y,                       Y_RANGE[0], Y_RANGE[1]),
    "reach_dist": (lambda s, t: math.sqrt(s.x**2+s.y**2),  0.0,        0.64),
    "orient":     (lambda s, t: float(s.orientation_id),   1.0,        5.0),
    "path_len":   (lambda s, t: path_length(t),            0.0,        3.0),
    "max_z":      (lambda s, t: max((p[2] for p in t), default=VOX_Z[0]), VOX_Z[0], VOX_Z[1]),
    "bbox_xy":    (lambda s, t: (
        (max(p[0] for p in t)-min(p[0] for p in t)) *
        (max(p[1] for p in t)-min(p[1] for p in t))) if t else 0.0, 0.0, 0.5),
}


def feature_value(name, scene, traj):
    return float(FEATURE_LIBRARY[name][0](scene, traj))


def feature_bin(name, value):
    _, fmin, fmax = FEATURE_LIBRARY[name]
    if fmax <= fmin:
        return 0
    idx = int((value - fmin) / (fmax - fmin) * GRID_SIZE)
    return max(0, min(GRID_SIZE - 1, idx))


STRUCTURAL_MAP = all(f in STRUCTURAL_FEATURES for f in MAP_FEATURES)


def scene_cell(scene):

    fx = feature_value(MAP_FEATURES[0], scene, [])
    fy = feature_value(MAP_FEATURES[1], scene, [])
    return (feature_bin(MAP_FEATURES[0], fx), feature_bin(MAP_FEATURES[1], fy))


# FAILURE-PROXIMITY PROXY

def compute_proxy(scene, trajectory):
    if not trajectory:
        return 10.0
    return min(math.hypot(p[0] - scene.x, p[1] - scene.y) for p in trajectory)


# DECISION-TREE FAILURE SURROGATE

class FailureSurrogate:

    def __init__(self):
        self.clf = None
        self.le = LabelEncoder().fit(list(OBJECTS.keys()))
        self.X = []
        self.y = []

    def _vec(self, scene):
        return [self.le.transform([scene.object_name])[0],
                scene.x, scene.y, scene.orientation_id]

    def add(self, scene, failure):
        self.X.append(self._vec(scene))
        self.y.append(int(failure))

    def fit(self):
        if len(set(self.y)) < 2:
            return False
        clf = DecisionTreeClassifier(max_depth=SURROGATE_MAX_DEPTH,
                                     class_weight="balanced")
        clf.fit(self.X, self.y)
        self.clf = clf
        return True

    def p_fail(self, scene):
        if self.clf is None:
            return 0.5
        proba = self.clf.predict_proba([self._vec(scene)])[0]
        classes = list(self.clf.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0

    @property
    def is_ready(self):
        return self.clf is not None

    def to_dataframe(self):
        rows = []
        for vec, fail in zip(self.X, self.y):
            obj_encoded = int(vec[0])
            rows.append({
                "object": self.le.inverse_transform([obj_encoded])[0],
                "object_encoded": obj_encoded,
                "x": vec[1], "y": vec[2], "ori_id": vec[3],
                "failure": int(fail),
            })
        return pd.DataFrame(rows, columns=["object", "object_encoded", "x", "y",
                                           "ori_id", "failure"])

    def save(self, base_dir):
        self.to_dataframe().to_csv(os.path.join(base_dir, "dt_dataset.csv"), index=False)
        if self.clf is not None:
            joblib.dump(self.clf, os.path.join(base_dir, "dt_model.joblib"))
            joblib.dump(self.le, os.path.join(base_dir, "dt_label_encoder.joblib"))

# INDIVIDUAL

class Individual:
    def __init__(self, scene, trajectory, failure):
        self.scene = scene
        self.trajectory = trajectory
        self.failure = failure
        self.features = {
            MAP_FEATURES[0]: feature_value(MAP_FEATURES[0], scene, trajectory),
            MAP_FEATURES[1]: feature_value(MAP_FEATURES[1], scene, trajectory),
        }
        self.cell = (feature_bin(MAP_FEATURES[0], self.features[MAP_FEATURES[0]]),
                     feature_bin(MAP_FEATURES[1], self.features[MAP_FEATURES[1]]))
        self.proxy = compute_proxy(scene, trajectory)
        self.new_cov = 0
        self.fitness = (1, 1e9)

    def set_fitness(self, new_cov_count):
        self.new_cov = new_cov_count
        fail_rank = 0 if self.failure == 1 else 1
        scalar = self.proxy - COVERAGE_LAMBDA * new_cov_count
        self.fitness = (fail_rank, scalar)

    def is_misbehaviour(self):
        return self.failure == 1

# SIMULATOR

def run_simulation(scene, eval_id, run_id, DATA_DIR, RESULTS_DIR, vla, model):
    scene_fname = f"t-grasp_n-1000_o-m3_s-2498586606_DH_HybridDT_Scenes_{eval_id}.json"
    scene_json_path = os.path.join(DATA_DIR, scene_fname)

    scene_data = {
        "0": {
            "model_id": OBJECTS[scene.object_name],
            "obj_init_options": {
                "init_xy": [scene.x, scene.y],
                "orientation": ORIENTATIONS[scene.orientation_id],
                "orientation_id": scene.orientation_id,
            },
        }
    }
    with open(scene_json_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    runner = SimRunner(vla, model, scene_json_path, RESULTS_DIR + "/")
    runner.run_exec()
    result_root = os.path.join(
        RESULTS_DIR, os.path.splitext(scene_fname)[0], model, "allMetrics", "0")

    trajectory = []
    tcp_file = os.path.join(result_root, "tcp_poses.json")
    if os.path.exists(tcp_file):
        with open(tcp_file, "r") as f:
            tcp_data = json.load(f)
        for pose in tcp_data:
            if isinstance(pose, list) and len(pose) >= 3:
                trajectory.append((pose[0], pose[1], pose[2]))
    if ALIGN_TRAJECTORY:
        trajectory = align_trajectory(trajectory)

    failure = 0
    log_file = os.path.join(result_root, "log.json")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
        last_key = str(len(logs) - 1)
        if last_key in logs:
            failure = 0 if logs[last_key]["success"] else 1

    return trajectory, failure

# Wilson lower bound

def wilson_lower_bound(successes, n, z=1.96):
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
    return (centre - margin) / denom

# DeepHyperion-VLA  (MAP-Elites + 3D coverage + surrogate-assisted selection)
def create_vla(model, task_type):
    instability_methods=['position_instability','velocity_instability', 'acceleration_instability']

    if "grasp" in task_type:
        vla = VLAInterface(model_name=model, task="google_robot_pick_customizable", instability_methods=instability_methods)
    elif "move" in task_type:
        vla = VLAInterface(model_name=model, task="google_robot_move_near_customizable",  instability_methods=instability_methods)
    elif "put-on" in task_type:
        vla = VLAInterface(model_name=model, task="widowx_put_on_customizable",  instability_methods=instability_methods)
    elif "put-in" in task_type:
        vla = VLAInterface(model_name=model, task="widowx_put_in_customizable",  instability_methods=instability_methods)
    else:
        raise NotImplementedError
    return vla
class DeepHyperionVLA:
    def __init__(self):
        self.map = {}
        self.cell_total = defaultdict(int)
        self.cell_fail = defaultdict(int)
        self.eval_count = 0
        self.spatial_covered = set()
        self.directional_covered = set()
        self.frontier = deque(maxlen=FRONTIER_SIZE)
        self.rows = []
        # --- surrogate state ---
        self.surrogate = FailureSurrogate()
        self.evaluated_genomes = []      # (x, y, ori_id) of every evaluated scene (ART pool)

    # ---- coverage metrics ----
    def spatial_coverage(self):
        return len(self.spatial_covered) / NUM_VOXELS if NUM_VOXELS else 0.0

    def directional_coverage(self):
        return len(self.directional_covered) / NUM_CONES if NUM_CONES else 0.0

    # ---- Evaluate ----
    def evaluate(self, scene, run_id, DATA_DIR, RESULTS_DIR, vla, model):
        eid = self.eval_count
        self.eval_count += 1
        MAX_RETRIES=10
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                traj, failure = run_simulation(
                    scene, eid, run_id, DATA_DIR, RESULTS_DIR, vla, model
                )
                break  # success → exit retry loop

            except Exception as e:
                print(f"[warn] attempt {attempt}/{MAX_RETRIES} failed for eval {eid}: {e}")

                if attempt == MAX_RETRIES:
                    print(f"[error] simulation permanently failed for eval {eid}")
                    return None
                time.sleep(5)

        ind = Individual(scene, traj, failure)
        pred_pfail = self.surrogate.p_fail(scene)
        vox, cones = trajectory_coverage(traj)
        new_vox = vox - self.spatial_covered
        new_cones = cones - self.directional_covered
        ind.set_fitness(len(new_vox) + len(new_cones))
        self.spatial_covered |= vox
        self.directional_covered |= cones

        if ind.new_cov > 0:
            self.frontier.append(ind)

        self.surrogate.add(scene, failure)
        self.evaluated_genomes.append((scene.x, scene.y, scene.orientation_id))

        self.rows.append({
            "eval_id": eid, "x": scene.x, "y": scene.y,
            "orientation_id": scene.orientation_id, "object_name": scene.object_name,
            "feat_x": ind.features[MAP_FEATURES[0]], "feat_y": ind.features[MAP_FEATURES[1]],
            "cell_x": ind.cell[0], "cell_y": ind.cell[1],
            "failure": failure, "proxy": ind.proxy, "new_coverage": ind.new_cov,
            "surrogate_pfail": pred_pfail,
            "surrogate_ready": int(self.surrogate.is_ready),
            "filled_cells": len(self.map), "mapped_misbehaviours": self.mapped_misbehaviours(),
            "spatial_coverage": self.spatial_coverage(),
            "directional_coverage": self.directional_coverage(),
        })
        return ind

    # ---- UpdateMap ----
    def update_map(self, ind):
        if ind is None:
            return
        c = ind.cell
        self.cell_total[c] += 1
        if ind.failure == 1:
            self.cell_fail[c] += 1
        cur = self.map.get(c)
        if cur is None or ind.fitness < cur.fitness:
            self.map[c] = ind

    # ---- metrics ----
    def mapped_misbehaviours(self):
        return sum(1 for ind in self.map.values() if ind.is_misbehaviour())

    def _sparseness(self, cells):
        if len(cells) < 2:
            return 0.0
        return sum(max(abs(i[0]-j[0])+abs(i[1]-j[1]) for j in cells) for i in cells) / len(cells)

    def misbehaviour_sparseness(self):
        return self._sparseness([c for c, ind in self.map.items() if ind.is_misbehaviour()])

    def coverage_sparseness(self):
        return self._sparseness(list(self.map.keys()))

    # ---- InitialisePopulation ----
    def select_diverse(self, seeds, k):
        if not seeds:
            return []
        chosen = [random.choice(seeds)]
        while len(chosen) < min(k, len(seeds)):
            best, best_d = None, -1
            for s in seeds:
                if s in chosen:
                    continue
                d = min(abs(s.cell[0]-c.cell[0])+abs(s.cell[1]-c.cell[1]) for c in chosen)
                if d > best_d:
                    best_d, best = d, s
            if best is None:
                break
            chosen.append(best)
        return chosen

    # SURROGATE-ASSISTED CANDIDATE SELECTION
    def _pick_parent(self):
        if self.frontier and random.random() < P_FRONTIER:
            return random.choice(self.frontier)
        if self.map:
            return self.map[random.choice(list(self.map.keys()))]
        return None

    def _candidate_novelty(self, scene):
        if not STRUCTURAL_MAP:
            return 0.5
        n = self.cell_total.get(scene_cell(scene), 0)
        return 1.0 / (1.0 + n)

    def _candidate_diversity(self, scene):
        if not self.evaluated_genomes:
            return 1.0
        gx = (X_RANGE[1] - X_RANGE[0]) or 1.0
        gy = (Y_RANGE[1] - Y_RANGE[0]) or 1.0
        dmin = 1e9
        for (ex, ey, eo) in self.evaluated_genomes:
            dx = (scene.x - ex) / gx
            dy = (scene.y - ey) / gy
            do = (scene.orientation_id - eo) / 4.0
            d = math.sqrt(dx*dx + dy*dy + do*do)
            if d < dmin:
                dmin = d
        return min(1.0, dmin)

    def _acquisition(self, scene):
        p_fail = self.surrogate.p_fail(scene) if USE_SURROGATE else 0.0
        novelty = self._candidate_novelty(scene)
        diversity = self._candidate_diversity(scene)
        return W_FAIL * p_fail + W_NOVELTY * novelty + W_DIVERSITY * diversity

    def propose_child(self):
        n = SURROGATE_CANDIDATES if USE_SURROGATE else 1
        pool = []
        for _ in range(n):
            parent = self._pick_parent()
            if parent is None:
                child = Scene()
            else:
                child = parent.scene.mutate()
                tries = 0
                while not child.differs_from(parent.scene) and tries < 10:
                    child = parent.scene.mutate()
                    tries += 1
            pool.append(child)

        if not USE_SURROGATE:
            return pool[0]

        # epsilon-greedy: forced exploration so the surrogate can't collapse diversity
        if random.random() < SURROGATE_EPSILON:
            return random.choice(pool)

        best, best_score = pool[0], -1e18
        for c in pool:
            s = self._acquisition(c)
            if s > best_score:
                best_score, best = s, c
        return best

    # ---- main loop ----
    def run(self, run_id, BASE_DIR, vla, model):
        DATA_DIR = os.path.join(BASE_DIR, "scene_generator")
        RESULTS_DIR = os.path.join(BASE_DIR, "results")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        print(f"[Run {run_id}] generating {SEED_POOL_SIZE} seeds ...")
        seeds = []
        pbar = tqdm(total=SEED_POOL_SIZE, desc="Evaluating seeds")

        for _ in range(SEED_POOL_SIZE):

            if self.eval_count >= EVAL_BUDGET:
                break

            ind = self.evaluate(Scene(), run_id, DATA_DIR, RESULTS_DIR, vla, model)

            if ind is not None:
                seeds.append(ind)
                self.update_map(ind)

            pbar.update(1)
            torch.cuda.empty_cache()  # clear GPU memory after each evaluation

        pbar.close()

        population = self.select_diverse(seeds, POP_SIZE)

        # first surrogate fit on the seed data (if both classes seen)
        if USE_SURROGATE and self.surrogate.fit():
            print(f"[Run {run_id}] surrogate trained on {len(self.surrogate.y)} seed evals")

        while self.eval_count < EVAL_BUDGET:

            child = self.propose_child()
            ind = self.evaluate(child, run_id, DATA_DIR, RESULTS_DIR, vla, model)
            self.update_map(ind)

            # periodic retrain on the growing dataset
            if USE_SURROGATE and (self.eval_count % SURROGATE_RETRAIN_EVERY == 0):
                self.surrogate.fit()

            sur = "on" if (USE_SURROGATE and self.surrogate.is_ready) else "off"
            print(f"[Run {run_id}] eval {self.eval_count}/{EVAL_BUDGET} "
                  f"| cells {len(self.map)} | misb {self.mapped_misbehaviours()} "
                  f"| spatial {self.spatial_coverage():.2%} "
                  f"| direction {self.directional_coverage():.2%} "
                  f"| surrogate {sur}")
            torch.cuda.empty_cache()


        self.save(BASE_DIR)

    # ---- persistence ----
    def save(self, BASE_DIR):
        pd.DataFrame(self.rows).to_csv(os.path.join(BASE_DIR, "dh_evaluations.csv"), index=False)

        # surrogate training data + final trained tree
        if USE_SURROGATE:
            self.surrogate.save(BASE_DIR)

        arch = []
        for cell, ind in sorted(self.map.items()):
            n, f = self.cell_total[cell], self.cell_fail[cell]
            mp = f / n if n else 0.0
            lb = wilson_lower_bound(f, n)
            arch.append({
                "cell_x": cell[0], "cell_y": cell[1],
                "obj_x": ind.scene.x, "obj_y": ind.scene.y,
                "orientation_id": ind.scene.orientation_id, "object_name": ind.scene.object_name,
                "fail_rank": ind.fitness[0], "scalar_fitness": ind.fitness[1],
                "new_coverage": ind.new_cov, "is_misbehaviour": ind.is_misbehaviour(),
                "n_inputs": n, "n_failures": f, "misbehaviour_prob": mp, "mp_ci_lower": lb,
                "high_failure_region": (mp > MP_THRESHOLD and lb > MP_CI_LOWER_THRESHOLD),
            })
        pd.DataFrame(arch).to_csv(os.path.join(BASE_DIR, "dh_archive.csv"), index=False)

        mp_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
        for cell in self.cell_total:
            n = self.cell_total[cell]
            mp_grid[cell[1], cell[0]] = (self.cell_fail[cell] / n) if n else 0.0
        np.savetxt(os.path.join(BASE_DIR, "dh_mp_grid.csv"), mp_grid, delimiter=",")

        summary = {
            "map_features": list(MAP_FEATURES), "grid_size": GRID_SIZE,
            "evaluations": self.eval_count, "filled_cells": len(self.map),
            "mapped_misbehaviours": self.mapped_misbehaviours(),
            "misbehaviour_sparseness": self.misbehaviour_sparseness(),
            "coverage_sparseness": self.coverage_sparseness(),
            "total_failures": int(sum(self.cell_fail.values())),
            "spatial_coverage": self.spatial_coverage(),
            "directional_coverage": self.directional_coverage(),
            "voxels_covered": len(self.spatial_covered), "total_voxels": NUM_VOXELS,
            "cones_covered": len(self.directional_covered), "total_cones": NUM_CONES,
            "high_failure_regions": sum(1 for a in arch if a["high_failure_region"]),
            "surrogate_used": USE_SURROGATE,
            "surrogate_candidates": SURROGATE_CANDIDATES,
            "surrogate_final_trained": (USE_SURROGATE and self.surrogate.is_ready),
        }
        with open(os.path.join(BASE_DIR, "dh_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== RUN SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            im = ax[0].imshow(mp_grid, origin="lower", cmap="Greys", vmin=0, vmax=1)
            ax[0].set_xlabel(MAP_FEATURES[0]); ax[0].set_ylabel(MAP_FEATURES[1])
            ax[0].set_title("Failure probability map (RQ3)")
            fig.colorbar(im, ax=ax[0], label="Misbehaviour Probability")
            df = pd.DataFrame(self.rows)
            ax[1].plot(df["eval_id"], df["spatial_coverage"], label="Spatial(T)")
            ax[1].plot(df["eval_id"], df["directional_coverage"], label="Directional(T)")
            ax[1].set_xlabel("evaluation"); ax[1].set_ylabel("coverage")
            ax[1].set_title("3D trajectory coverage growth"); ax[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, "dh_maps.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"  (plots skipped: {e})")

# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation configuration")
    parser.add_argument("--budget",type=int,required=False,help="Evaluation budget (int)", default=400)
    parser.add_argument("--model",type=str,required=False,help="Model name (str)", default="openvla")
    parser.add_argument("--task",type=str,required=False,help="Task type (str)", default="grasp")
    args = parser.parse_args()
    EVAL_BUDGET = args.budget
    base_root = f"../data/DH_HybridDT_{args.budget}scenes"
    model=args.model
    task_type=args.task
    vla=create_vla(model, task_type)
    for r in range(1, NUM_RUNS + 1):
        print(f"\n===== RUN {r} =====")
        BASE_DIR = os.path.join(base_root, f"run_{r}", model)
        os.makedirs(BASE_DIR, exist_ok=True)
        DeepHyperionVLA().run(r, BASE_DIR, vla, model)
    print("\nAll runs completed!")
