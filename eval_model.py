import os

import numpy as np
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds

NUM_TIMES = 2#100
# CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
#         [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
#         [128, 40, 0.15], [128, 40, 0.3]]

CASE = [[128, 40, 0.0], [128, 30, 0], [128, 30, 0.15], [128, 30, 0.3], [128, 50, 0.3], [128, 50, 0.15],[128, 50, 0]]

set_global_seeds(SetupParameters.SEED)


def one_step(env0, num_agent, actions, predicted_action, comm_mask, curr_pos_time, model0, pre_value, input_state, ps, one_episode_perf, message, episodic_buffer0):
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, num_collide, _, modify_actions = env0.joint_step(
        actions, predicted_action, comm_mask, curr_pos_time, one_episode_perf['episode_len'], model0, pre_value, input_state, ps, no_reward=False, message=message,
        episodic_buffer=episodic_buffer0)

    one_episode_perf['collide'] += num_collide
    vector[:, :, -1] = modify_actions
    one_episode_perf['episode_len'] += 1

    t_actions = {}
    
    # actions is a dictionary with ids as agent's index..
    for t in range(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT):
        t_actions[t] = []
        # for agent in range(num_agent):
            # greedy_action = np.argmax(policy, axis=-1)
        for i in range(num_agent):
            # print(i)
            # if not greedy:
            eval_action = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        # if greedy:
            # eval_action = greedy_action
            t_actions[t].append(eval_action)
            
    coll_mask, curr_pose_time, img_state_time  = env.Agents_Collision_Check(t_actions, num_agent) # contains index of agents which collide
    actions_n = convert_to_numpy(t_actions, num_agent)
    curr_pose_time = np.reshape(curr_pose_time, (1, num_agent, 2*(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT+1)))
    
    return reward, obs, vector, done, one_episode_perf, max_on_goal, on_goal, coll_mask, curr_pose_time, img_state_time, actions_n


def convert_to_numpy(actions, num_agent):
    agents_act = []
    for time in range(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT):
        agents_action = actions[time]
        agents_act.append(np.array(agents_action, dtype=np.float32))

    actions_n = np.reshape(np.concatenate(agents_act, 0), (1, num_agent, TrainingParameters.PREDICTION_TIMESTEPS_LIMIT))

    return actions_n

def evaluate(eval_env, model0, device, episodic_buffer0, num_agent, save_gif0):
    """Evaluate Model."""
    one_episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0}
    episode_frames = []

    done, _, obs, vector, _, coll_agent = reset_env(eval_env, num_agent)

    episodic_buffer0.reset(2e6, num_agent)
    new_xy = eval_env.get_positions()
    episodic_buffer0.batch_add(new_xy)

    message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(torch.device('cpu'))
    hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                    torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))

    curr_pose_time = np.zeros((1, num_agent, (TrainingParameters.PREDICTION_TIMESTEPS_LIMIT+1)*2), dtype=np.float32)
    predicted_actions = np.zeros((1, num_agent, TrainingParameters.PREDICTION_TIMESTEPS_LIMIT), dtype=np.float32)
    
    if save_gif0:
        episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))
    comm_mask_list = []
    while not done:
        actions, hidden_state, v_all, ps, message = model0.final_evaluate(obs, vector, hidden_state, message, num_agent,
                                                                          False, coll_agent, predicted_actions, curr_pose_time)

        rewards, obs, vector, done, one_episode_perf, max_on_goals, on_goal, coll_agent, curr_pose_time, img_state_time, predicted_actions = one_step(eval_env, num_agent, actions, \
                                                                                       predicted_actions, coll_agent, curr_pose_time, model0, v_all,
                                                                                       hidden_state, ps,
                                                                                       one_episode_perf, message,
                                                                                       episodic_buffer0)
        new_xy = eval_env.get_positions()
        comm_mask_list.append(coll_agent)
        processed_rewards, _, intrinsic_reward, min_dist = episodic_buffer0.if_reward(new_xy, rewards, done, on_goal)

        vector[:, :, 3] = rewards
        vector[:, :, 4] = intrinsic_reward
        vector[:, :, 5] = min_dist

        if save_gif0:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        if done:
            if one_episode_perf['episode_len'] < EnvParameters.EPISODE_LEN - 1:
                one_episode_perf['success_rate'] = 1
            one_episode_perf['max_goals'] = max_on_goals
            one_episode_perf['collide'] = one_episode_perf['collide'] / (
                    (one_episode_perf['episode_len'] + 1) * num_agent)
            if save_gif0:
                if not os.path.exists(RecordingParameters.GIFS_PATH):
                    os.makedirs(RecordingParameters.GIFS_PATH)
                images = np.array(episode_frames)
                make_gif(images, '{}/evaluation.gif'.format(
                    RecordingParameters.GIFS_PATH))
        break
    
    return one_episode_perf, comm_mask_list


if __name__ == "__main__":
    # download trained model0
    # model_path = './models/MAPF_future/SCRIMP_+_future20-03-230914/final'
    model_path = '.'
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    model = Model(0, torch.device('cpu'))
    model.network.load_state_dict(torch.load(path_checkpoint)['model'])
    summary_path="./"
    txt_path = summary_path + '/' + RecordingParameters.TXT_NAME
    save_gif = False#True
    # start evaluation
    for k in CASE:
        # remember to modify the corresponding code (size,prob) in the 'mapf_gym.py'
        comm_per_case = []

        env = MAPFEnv(num_agents=k[0], size=k[1], prob=k[2])
        episodic_buffer = EpisodicBuffer(2e6, k[0])

        all_perf_dict = {'episode_len': [], 'max_goals': [], 'collide': [], 'success_rate': []}
        all_perf_dict_std = {'episode_len': [], 'max_goals': [], 'collide': []}
        print('agent: {}, world: {}, obstacle: {}'.format(k[0], k[1], k[2]))

        for j in range(NUM_TIMES):
            eval_performance_dict, comm_mask_list = evaluate(env, model, torch.device('cpu'), episodic_buffer, k[0], save_gif)
            comm_per_case.append(comm_mask_list)
            save_gif = False  # here we only record gif once
            if j % 20 == 0:
                print(j)

            for i in eval_performance_dict.keys():  # for one episode
                if i == 'episode_len':
                    if eval_performance_dict['success_rate'] == 1:
                        all_perf_dict[i].append(eval_performance_dict[i])  # only record success episode
                    else:
                        continue
                else:
                    all_perf_dict[i].append(eval_performance_dict[i])

        for i in all_perf_dict.keys():  # for all episodes
            if i != 'success_rate':
                all_perf_dict_std[i] = np.std(all_perf_dict[i])
            all_perf_dict[i] = np.nanmean(all_perf_dict[i])


        print('EL: {}, MR: {}, CO: {},SR:{}'.format(round(all_perf_dict['episode_len'], 2),
                                                    round(all_perf_dict['max_goals'], 2),
                                                    round(all_perf_dict['collide'] * 100, 2),
                                                    all_perf_dict['success_rate'] * 100))
        print('EL_STD: {}, MR_STD: {}, CO_STD: {}'.format(round(all_perf_dict_std['episode_len'], 2),
                                                          round(all_perf_dict_std['max_goals'], 2),
                                                          round(all_perf_dict_std['collide'] * 100, 2)))
        print('-----------------------------------------------------------------------------------------------')

        # with open(txt_path, "w") as f:
        #     f.write(str('agent: {}, world: {}, obstacle: {} \n EL: {}, MR: {}, CO: {},SR:{}, EL_STD: {}, MR_STD: {}, CO_STD: {}'.format(k[0], k[1], k[2],
        #                                                 round(all_perf_dict['episode_len'], 2),
        #                                                 round(all_perf_dict['max_goals'], 2),
        #                                                 round(all_perf_dict['collide'] * 100, 2),
        #                                                 all_perf_dict['success_rate'] * 100,
        #                                                 round(all_perf_dict_std['episode_len'], 2),
        #                                                 round(all_perf_dict_std['max_goals'], 2),
        #                                                round(all_perf_dict_std['collide'] * 100, 2))))        
    
        title = "./masking/"+str(k)+".npy"
        np.save(title, comm_per_case)

   
    
    print('finished')
