import gymnasium
from comet_ml import Experiment
import cv2
import os

class RolloutVideoWrapper(gymnasium.Wrapper):
    def __init__(self, env, logger, video_folder = "logs/skrl/videos",
                 episode_frequency = 100, recorded_env = 0):
        super().__init__(env)
        os.makedirs(f'{os.getcwd()}/{video_folder}', exist_ok=True)
        self.logger = logger
        self.video_folder = video_folder
        self.episode_frequency = episode_frequency
        self.current_episode = -1
        self.video_buffer = []
        self.recorded_env = recorded_env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated[self.recorded_env] or truncated[self.recorded_env]:
            if len(self.video_buffer) > 0:
                video_path = f"{self.video_folder}/episode_{self.current_episode}.mp4"
                self.log_video()
                self.video_buffer = []
            self.current_episode += 1

        if self.current_episode % self.episode_frequency == 0:
            self.video_buffer.append(self.env.render()[0])

        return obs, reward, terminated, truncated, info
    
    def reset(self):
        return self.env.reset()
    
    def log_video(self):
        video_path = f"{self.video_folder}/episode_{self.current_episode}.mp4"
        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (self.video_buffer[0].shape[2], self.video_buffer[0].shape[1])
        )
        for frame in self.video_buffer:
            video.write(frame.permute(1, 2, 0).cpu().numpy())
        video.release()

        if isinstance(self.logger, Experiment):
            video_path = f"{os.getcwd()}/{self.video_folder}/episode_{self.current_episode}.mp4"
            self.logger.log_video(
                file = video_path,
                name=f"episode_examples",
                step=self.current_episode,
                format="mp4"
            )
            print("video logged")
            self.video_buffer = []
            os.remove(video_path)
