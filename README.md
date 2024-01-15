# Swedish_HTR

## This repo is for the course Project in Data Science.

#### This project is based on Loghi HTR workflow and transfer learning.
## Usage
1. Pull this repo
2. Download all models provided in the link below (best_nov_30 and laypa model). best_nov_30 is HTR model for transcribing, it is a folder while laypa model is used for baseline detection, it contains two files (one pth file and one yaml conf file).
[best_nov_30](https://1drv.ms/f/s!AhLc1l9ln_UugvBxsrxiS2OCpx5f3w?e=NjdVju)
[laypa model](https://1drv.ms/f/s!AhLc1l9ln_UugvEF1UOXzfOyDoWtwA?e=CjMQWx)
3. Downlaod 3 Apptainer images (laypa.sif, htr.sif, tool.sif).
[images](https://1drv.ms/f/s!AhLc1l9ln_UugvEGg39WPvRkaehvdg?e=lnt6hl)
4. Edit certain directories in shell scripts like apptainer-pipe.sh into model files in your system.
5. To try the model, use apptainer-pipe.sh and put images you want to process into the *image* folder
6. Run apptainer-pipe.sh, and after running it, you should expect to see a *page* folder inside *image* folder, where you can find pairs of baseline detection results and transcribing results.