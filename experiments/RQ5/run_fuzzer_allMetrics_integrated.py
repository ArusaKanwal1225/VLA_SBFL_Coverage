"""
Name   : run_fuzzer_allMetrics.py
Author : PABLO VALLE
Time   : 8/8/24
"""
import argparse
import numpy as np
from model_interface_allMetrics import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
import shutil
import cv2
import re
import torch
import subprocess


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return bool(obj) if isinstance(obj, np.bool_) else super().default(obj)
class SimRunner:
    def __init__(self, vla, model_name, data, output, image_output=None, seed=None, resume=None):
        self.vla=vla
        self.model=model_name
        self.data=data
        self.output=output

        self.image_output=image_output
        self.seed=seed
        self.resume=resume
    def run_exec(self, task_type_override=None, prompts=None, result_dir_override=None,
                 episode_callback=None, write_video=True):

        random_seed = self.seed if self.seed is not None else np.random.randint(0, 4294967295)  # max uint32

        #task_data="t-grasp_n-1000_o-m3_s-2498586606_redbull_20_test9.json"
        #task_data="t-grasp_n-1000_o-m3_s-2498586606_scenes_fscs_art_dt_online_100.json"

        task_data="data/Hybrid_2_FSCSARTDT_500scenes/scene_generator/t-grasp_n-1000_o-m3_s-2498586606_Hybrid_FSCSARTDT_500_scene_{i}.json"
        data_path = self.data if self.data else str(PACKAGE_DIR) + "/../data/Hybrid_2_FSCSARTDT_500scenes/scene_generator/"+task_data
        dataset_name = data_path.split('/')[-1]

        match = re.search(r't-(.*?)_n', dataset_name)

        if task_type_override is not None:
            task_type = task_type_override
        elif match:
            task_type = match.group(1)
            #print(task_type)  # Output: put-in
        else:
            print("No match found")



        with open(data_path, 'r') as f:
            tasks = json.load(f)

        if result_dir_override is not None:
            result_dir = str(result_dir_override)
        elif self.output:
            result_dir = self.output + data_path.split('/')[-1].split(".")[0]
        else:
            result_dir = str(PACKAGE_DIR) + "/../results/" + data_path.split('/')[-1].split(".")[0]
        os.makedirs(result_dir, exist_ok=True)
        if result_dir_override is None:
            result_dir += f'/{self.model}'#_{random_seed}'
        if not self.resume and result_dir_override is None:
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)


        #if args.image_output:
        #    image_dir = args.image_output + data_path.split('/')[-1].split(".")[0]
        #    os.makedirs(image_dir, exist_ok=True)
        #    image_dir += f'/{args.model}_{random_seed}'
        #    os.makedirs(image_dir, exist_ok=True)
        #else:
        #    image_dir = None


        for idx in (sorted(tasks, key=int) if prompts is not None else tasks):
            if self.resume and os.path.exists(result_dir + f"/allMetrics/{idx}/" + '/log.json'):
                continue
            options = tasks[str(idx)]
            prompt = prompts[int(idx)] if prompts is not None else None
            images, episode_stats, actions, tcp_poses, uncertainty_token, uncertainty_variability, optimal_traj, traj_inst_gradients,  traj_instability, traj_instability_tcp, exec_times_dict  = self.vla.run_interface(seed=random_seed, options=options, task_type=task_type, prompt=prompt)
            os.makedirs(result_dir + f"/allMetrics/{idx}", exist_ok=True)


    # ----------------------------- UNCERTAINTY METRICS ----------------------------------------------------------------------------------------------------
            #Execution time
            for method_name, values in exec_times_dict.items():
                file_path = result_dir + f"/allMetrics/{idx}/" + f"exec_time_{method_name}.json"
                with open(file_path, "w") as f:
                    json.dump(self.convert_to_native(values), f, indent=2)


            # Token-based metrics
            for method_name, values in uncertainty_token.items():
                file_path = result_dir + f"/allMetrics/{idx}/" + f"Token_based_{method_name}.json"
                with open(file_path, "w") as f:
                    json.dump(self.convert_to_native(values), f, indent=2)



            # Execution Variability
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in uncertainty_variability]
            file_path = result_dir + f"/allMetrics/{idx}/" + f"Execution_Variability.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)


            # Optimal trejectory
            optimal_traj=np.array(optimal_traj).tolist()
            uncertainty = np.diff(optimal_traj, prepend=optimal_traj[0]) if optimal_traj else np.array([])
            normalized_uncertainty = ((uncertainty + 1) / 2).tolist()
            with open(result_dir + f"/allMetrics/{idx}/" + '/Optimal_Trajectory.json', "w") as f:
                json.dump(normalized_uncertainty, f, indent=4)


            # Trajectory Instability Gradients
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in traj_inst_gradients]
            file_path = result_dir + f"/allMetrics/{idx}/" + f"Trajectory_Instability_Gradients.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)


            # Trajectory Instability
            for method_name, values in traj_instability.items():
                serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
                file_path = result_dir + f"/allMetrics/{idx}/" + f"{method_name}.json"
                with open(file_path, "w") as f:
                    json.dump(serialized_values, f, indent=4)

            for method_name, values in traj_instability_tcp.items():
                serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
                file_path = result_dir + f"/allMetrics/{idx}/" + f"TCP_{method_name}.json"
                with open(file_path, "w") as f:
                    json.dump(serialized_values, f, indent=4)


    #-------------------------------------------------------------------------------------------------------------------------------------------------------


            #print(actions)
            json_ready_actions=[
                {key: self.convert_to_native(value) for key, value in entry.items()}
                for entry in actions
            ]
            with open(result_dir + f"/allMetrics/{idx}/" + '/actions.json', "w") as f:
                json.dump(json_ready_actions, f, indent=2)

            with open(result_dir + f"/allMetrics/{idx}/" + '/tcp_poses.json', "w") as f:
                json.dump(tcp_poses, f, indent=2)
            #if image_dir:
                #os.makedirs(image_dir + f"/{idx}", exist_ok=True)

            if write_video:
                video_path = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation_orig.mp4")
                video_path_dest = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation.mp4")

                # Get frame size from the first image
                height, width = images[0].shape[:2]

                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
                fps = 10  # Set your desired FPS
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                imagePath=os.path.join(result_dir , f"allMetrics/{idx}","images")
                os.makedirs(imagePath,exist_ok=True)
                for img_idx in range(len(images)):
                    frame = images[img_idx]

                    # Ensure frame is uint8 and in BGR (OpenCV uses BGR not RGB)
                    if frame.dtype != np.uint8:
                        frame = (255 * (frame - frame.min()) / (frame.ptp() + 1e-8)).astype(np.uint8)

                    if frame.shape[2] == 3:  # RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    out.write(frame)
                    #im = Image.fromarray(images[img_idx])
                    #im.save(imagePath+"/Input_"+str(img_idx)+".jpg")



                out.release()
                command = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-c:v", "h264",  # Or "openh264" if needed
                    video_path_dest
                ]

                # Run the command
                try:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    #print(f"Video saved to {video_path_dest}")
                except subprocess.CalledProcessError as e:
                    print("Error during conversion:", e)
                os.remove(video_path)
                #print(f"Video saved to {video_path}")
                #for img_idx in range(len(images)):
                #    print(len(images[img_idx]))
                #    im = Image.fromarray(images[img_idx])
                #    im.save(result_dir + f"/{idx}/" + f'{img_idx}.j


            with open(result_dir + f"/allMetrics/{idx}/" + '/log.json', "w") as f:
                json.dump(episode_stats, f, cls=StableJSONizer)
            if episode_callback is not None:
                episode_callback(str(idx), options, prompt, episode_stats, tcp_poses)

    def convert_to_native(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, list):
            return [self.convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_to_native(v) for k, v in obj.items()}
        return obj
