from tensorboard import program

tracking_address = r'D:\projects\RL\highway-env\scripts\highway_ppo'

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input('listening')