import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import numpy as np

def events_to_combined_video_normalized(npzfile1, npzfile2, frame_size=(34, 34), num_frames=100, filename="normalized_combined_events.mp4", fps=10):
    """
    Create a combined video from two unsynchronized event sequences by normalizing timestamps.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        npzfile2 (str): Path to the second .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
        num_frames (int): Total number of frames to render.
        filename (str): Output combined video filename.
        fps (int): Frames per second for the video.
    """
    # Load the .npz files
    # data1 = np.load(npzfile1)
    # data2 = np.load(npzfile2)
    data1 = npzfile1
    data2 = npzfile2
    
    # Extract events from each file
    x1, y1, t1, p1 = data1['x'], data1['y'], data1['t'], data1['p']
    x2, y2, t2, p2 = data2['x'], data2['y'], data2['t'], data2['p']
    
    # Combine into lists of tuples (x, y, t, p) for each sequence
    events1 = list(zip(x1, y1, t1, p1))
    events2 = list(zip(x2, y2, t2, p2))
    
    # Find the min and max times for normalization
    min_time1, max_time1 = min(t1), max(t1)
    min_time2, max_time2 = min(t2), max(t2)
    
    # Overall min and max times across both sequences
    min_time = min(min_time1, min_time2)
    max_time = max(max_time1, max_time2)
    
    # Normalize timestamps to a range of 0 to num_frames-1
    def normalize_time(events, min_time, max_time, num_frames):
        return [(x, y, int((t - min_time) / (max_time - min_time) * (num_frames - 1)), p) for x, y, t, p in events]

    events1 = normalize_time(events1, min_time, max_time, num_frames)
    events2 = normalize_time(events2, min_time, max_time, num_frames)
    
    # Create a video writer
    writer = imageio.get_writer(filename, fps=fps)
    
    # Initialize accumulated frames for both sequences
    accumulated_frame1 = np.zeros(frame_size, dtype=float)
    accumulated_frame2 = np.zeros(frame_size, dtype=float)
    
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
        
        # Process events for the current frame index for the second sequence
        # for event in events2:
        #     x, y, t, p = event
        #     if t == frame_idx:  # Match the current normalized time frame
        #         frame2[y, x] += 1 if p > 0 else -1
        while pt2 < len(events2) and events2[pt2][2] < frame_idx:
            pt2 += 1
        while pt2 < len(events2) and events2[pt2][2] == frame_idx:
            x, y, t, p = events2[pt2]
            ### frame2[y, x] += 1 if p > 0 else -1
            frame2[y, x] += p
            pt2 += 1
        
        # Normalize both frames for visualization
        ## normalized_frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-6)
        ## normalized_frame2 = (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-6)
        
        # Apply a colormap for better contrast and informative visualization
        img1 = plt.cm.viridis(frame1) * 255  # Apply plasma colormap and scale to 0-255
        img2 = plt.cm.viridis(frame2) * 255  # Apply plasma colormap and scale to 0-255
        
        # Remove alpha channel by taking only RGB values (0:3)
        img1 = img1[:, :, :3].astype(np.uint8)
        img2 = img2[:, :, :3].astype(np.uint8)
        
        # Concatenate the two frames side-by-side
        combined_frame = np.hstack((img1, img2))
        
        # Append combined frame to video
        writer.append_data(combined_frame)
    
    writer.close()
    print(f"Combined video saved as {filename}")

# Example usage
# events_to_combined_video_normalized("path_to_file1.npz", "path_to_file2.npz", frame_size=(34, 34), num_frames=100, filename="normalized_combined_events.mp4", fps=10)
