from render import *
import numpy as np
import matplotlib.pyplot as plt

npz_1 = np.load("/ailab/user/lixinyu/event-data/DVSGesture/events_np/train/1/user01_fluorescent_led_0.npz")
npz_2 = np.load("/ailab/user/lixinyu/event-data/DVSGesture/events_np/train/1/user06_fluorescent_led_0.npz")
print(dict(npz_1))
print(dict(npz_2))

n_h, n_w = 6, 8

## events_to_combined_video_normalized(npz_1, npz_2, frame_size=(128, 128))

def create_image_grid(images, grid_size=(8, 8)):
    # Reshape the list into a grid format
    rows = []
    for i in range(grid_size[0]):
        row = np.concatenate(images[i * grid_size[1]:(i + 1) * grid_size[1]], axis=1)  # Concatenate along width
        rows.append(row)
    grid_image = np.concatenate(rows, axis=0)  # Concatenate along height
    
    return grid_image

def events_to_pics(npzfile1, frame_size=(34, 34), filename="pics.jpg", num_frames=100, fps=10):
    """
    Create a combined video from two unsynchronized event sequences by normalizing timestamps.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
    """
    # Load the .npz files
    # data1 = np.load(npzfile1)
    # data2 = np.load(npzfile2)
    data1 = npzfile1
    
    # Extract events from each file
    x1, y1, t1, p1 = data1['x'], data1['y'], data1['t'], data1['p']
    
    # Combine into lists of tuples (x, y, t, p) for each sequence
    events1 = list(zip(x1, y1, t1, p1))
    
    # Find the min and max times for normalization
    min_time1, max_time1 = min(t1), max(t1)
    
    # Overall min and max times across both sequences
    # min_time = min(min_time1, min_time2)
    # max_time = max(max_time1, max_time2)
    min_time = min_time1
    max_time = max_time1
    
    # Normalize timestamps to a range of 0 to num_frames-1
    def normalize_time(events, min_time, max_time, num_frames):
        return [(x, y, int((t - min_time) / (max_time - min_time) * (num_frames - 1)), p) for x, y, t, p in events]

    events1 = normalize_time(events1, min_time, max_time, num_frames)
    
    print(events1[:5])
    print('......')
    print(events1[-3:])
    
    # Create a video writer
    writer = imageio.get_writer(filename.replace('.jpg', '.mp4'), fps=fps)
    
    # num = 6
    # wanted_frame_indices = np.arange(0, num_frames, num_frames//20).tolist()[:num]
    imgs = []
    
    pt1, pt2 = 0, 0
    # Generate frames and write to video

    for frame_idx in tqdm(range(num_frames), desc="Generating normalized frames"):
        # Clear frames for current frame
        frame1 = np.zeros(frame_size, dtype=float)
        frame2 = np.zeros(frame_size, dtype=float)
        
        # Process events for the current frame index for the first sequence
        # for event in events1:
        #     x, y, t, p = event
        #     if t == frame_idx:  # Match the current normalized time frame
        #         frame1[y, x] += 1 if p > 0 else -1
        while pt1 < len(events1) and events1[pt1][2] < frame_idx:
            pt1 += 1
        while pt1 < len(events1) and events1[pt1][2] == frame_idx:
            x, y, t, p = events1[pt1]
            frame1[y, x] += p
            pt1 += 1
        
        # Normalize both frames for visualization
        normalized_frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-6)
        
        # Apply a colormap for better contrast and informative visualization
        img1 = plt.cm.inferno(normalized_frame1) * 255  # Apply inferno colormap and scale to 0-255
        
        # Remove alpha channel by taking only RGB values (0:3)
        img1 = img1[:, :, :3].astype(np.uint8)
        
        # Append combined frame to video
        ## if frame_idx in wanted_frame_indices:
        if frame_idx < n_h * n_w:
            imgs.append(img1)
        writer.append_data(img1)
    
    ## writer.close()
    ## print(f"Combined video saved as {filename}")
    
    writer.close()
    
    # add some minor white spaces as separators
    # seps = [np.ones((128, 2, 3), dtype=int) * 255 for i in range(num-1)]
    # _tmp = []
    # for _ in range(num-1):
    #     _tmp += [imgs[_]]
    #     _tmp += [seps[_]]
    # _tmp += [imgs[-1]]

    
    # out_img = np.concatenate(_tmp, axis=1)
    out_img = create_image_grid(imgs, (n_h, n_w))
    
    fig = plt.figure()
    plt.imshow(out_img, cmap='inferno')
    plt.axis("off")
    ## plt.colorbar()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
if __name__ == '__main__':
    npz_1 = np.load("/ailab/user/lixinyu/event-data/DVSGesture/events_np/train/5/user02_fluorescent_led_0.npz")
    events_to_pics(npz_1, frame_size=(128, 128))