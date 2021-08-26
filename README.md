# CuttingFace
Data maker tool, cutting face from video

# Inatall 
- Insightfaces version 0.4.1
- [Download](https://1drv.ms/u/s!AswpsDO2toNKrU0ydGgDkrHPdJ3m?e=iVgZox) antelope weight

# Run
- Read help infomation: `python video_detector.py --help`
- Example to run sample.mp4 video, store face image to out_dir with file name 01_xx.jpg
  
  `python video_detector.py -v sample.mp4 -s out_dir -p 01`
 
- Press and hold any key, suggest key 'n' like 'next' :D, to show the frame on video
- At the frame contain face you decide to cut, press 'c' (sometimes it fail beacause waiting frame, try again)
- After all, review the face on directory that you choose to store
