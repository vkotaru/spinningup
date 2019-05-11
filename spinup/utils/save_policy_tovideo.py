import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import gym, gym_cassie
import numpy 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def run_policy(env, get_action, max_ep_len=2000, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    action_array = None
    list_actions = []
    while n < num_episodes:
        print("episode length: {}".format(ep_len))
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if action_array is None:
            action_array = numpy.reshape(a, (1,10))
        else:
            action_array = numpy.append(action_array, numpy.reshape(a, (1,10)), axis=0)

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

            list_actions.append(action_array)
            action_array = None

    plot_actions(list_actions)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def plot_actions(list_actions):
    
    plt.figure()
    for i in range(10):
        print("i:%d"% (i+1))
        ax1 = plt.subplot(2,5,i+1)
        for l in list_actions:
            ax1.plot(list(range(len(l[:,i]))), l[:,i], '-',linewidth=2.0)
        # plt.axis("equal")
        if (int(i/5) <1) :
            title_name='left'
        else:
            title_name='right'
        plt.title(title_name+'Motor: '+str((i%5)+1))
        ax1.set_ylim([-150,150])
        ax1.set_xlim([0,len(l[:,i]) ])
        ax1.grid()

        # plt.tight_layout()
    # ip.embed()
    plt.savefig('torques.png')
    plt.show()
    return
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=200)
    parser.add_argument('--episodes', '-n', type=int, default=3)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--env_name', type=str, default='Cassie-v1')
    args = parser.parse_args()
    _, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)

    # print(args.env_name)    
    env = gym.make(args.env_name)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))