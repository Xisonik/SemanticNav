import gymnasium
from comet_ml import Experiment
import cv2
import os

class RolloutVideoWrapper(gymnasium.Wrapper):
    def __init__(self, env, logger, video_folder = "logs/skrl/videos",
                 episode_frequency = 100):
        super().__init__(env)
        os.makedirs(f'{os.getcwd()}/{video_folder}', exist_ok=True)
        self.logger = logger
        self.video_folder = video_folder
        self.episode_frequency = episode_frequency
        self.current_episode = 0
        self.video_buffer = {'third': [], 'first': []}
        self.recorded_env = env.unwrapped.get_environment_which_is_closest_to_camera_lookat()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (self.current_episode % self.episode_frequency == 0) or\
                (self.current_episode % self.episode_frequency == (self.episode_frequency - 1) and\
                (terminated[self.recorded_env] or truncated[self.recorded_env])):
            self.video_buffer['first'].append(self.env.unwrapped.render_fpv()[self.recorded_env])
            self.video_buffer['third'].append(self.env.render())

        if terminated[self.recorded_env] or truncated[self.recorded_env]:
            if len(self.video_buffer['third']) > 1:
                self.log_video()
            self.current_episode += 1


        return obs, reward, terminated, truncated, info
    
    def reset(self):
        return self.env.reset()
    
    def log_video(self):
        for key in self.video_buffer.keys():
            video_path = f"{self.video_folder}/episode_{self.current_episode}_{key}.mp4"
            video = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (self.video_buffer[key][0].shape[1], self.video_buffer[key][0].shape[0])
            )
            for frame in self.video_buffer[key]:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video.release()

            if isinstance(self.logger, Experiment):
                video_path = f"{os.getcwd()}/{self.video_folder}/episode_{self.current_episode}_{key}.mp4"
                self.logger.log_video(
                    file = video_path,
                    name=f"episode_examples_{key}",
                    step=self.current_episode
                )
                self.video_buffer[key] = []
                os.remove(video_path)