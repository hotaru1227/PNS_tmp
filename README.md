## 指令：

1. Train the prompter

   ```shell
   python main.py --config pannuke123.py --output_dir pannuke123
   python main.py --config pannuke213.py --output_dir pannuke213
   python main.py --config pannuke321.py --output_dir pannuke321_change_feats3

   python main.py --config cpm17.py --output_dir cpm17/test_output_image
   ```

2. 生点

   ```shell
   python predict_prompts.py --config pannuke123.py --resume checkpoint/pannuke123/best.pth
   python predict_prompts.py --config pannuke213.py --resume checkpoint/pannuke213/best.pth
   python predict_prompts.py --config pannuke321.py --resume checkpoint/pannuke321/best.pth
   python predict_prompts.py --config cpm17.py --resume checkpoint/cpm17/test_output_image/best.pth
   ```

3. train the segmentor
   
   Download SAM's pre-trained [weights](https://github.com/facebookresearch/segment-anything) into **segmentor/pretrained** .

   ```shell
   cd segmentor
   torchrun --nproc_per_node=4 main.py --config pannuke123_b.py --output_dir pannuke123_b
   python main.py --config pannuke123_b.py --output_dir pannuke123_b
   torchrun --nproc_per_node=4 main.py --config pannuke213_b.py --output_dir pannuke213_b
   torchrun --nproc_per_node=4 main.py --config pannuke321_b.py --output_dir pannuke321_b
   ```

4. test（evaluation）

   ```shell
   cd segmentor
   torchrun --nproc_per_node=4 main.py --resume checkpoint/cpm17/cpm17_b.pth --eval --config cpm17_b.py --output_path /data/hotaru/projects/PNS_tmp/segmentor/outputAndOtherfile/cpm_test_del/
   python main.py --resume checkpoint/cpm17/latest.pth --eval --config cpm17_b.py
   python main.py --resume checkpoint/pannuke321_b/latest.pth --eval --config pannuke321_b.py

   ```
   



## Dataset_path

1. run **extract_data.py** for data pre-processing. 

   ```markdown
   datasets
   ├── cpm17
   │   ├── extract_data.py
   │   ├── test
   │   └── train
   ├── cpm17_test_files.npy
   ├── cpm17_train_files.npy
   ├── kumar
   │   ├── extract_data.py
   │   ├── images
   │   └── labels
   ├── kumar_test_files.npy
   ├── kumar_train_files.npy
   ├── pannuke
   │   ├── extract_data.py
   │   ├── Fold 1
   │   ├── Fold 2
   │   ├── Fold 3
   │   ├── Images
   │   └── Masks
   ├── pannuke123_test_files.npy
   ├── pannuke123_train_files.npy
   ├── pannuke123_val_files.npy
   ├── pannuke213_test_files.npy
   ├── pannuke213_train_files.npy
   ├── pannuke213_val_files.npy
   ├── pannuke321_test_files.npy
   ├── pannuke321_train_files.npy
   └── pannuke321_val_files.npy
   ```


## Checkpoints

|             |                            Kumar                             |                            CPM-17                            |                          PanNuke123                          |                          PanNuke213                          |                          PanNuke321                          |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Prompter   | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ee_5mPeYZIhGufpsumWbp1QBWPKLg6BxLoXoOzl9BGywVw?e=mHm8Wg) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ec0xXaiuz2JIjDInHq1tuEwBJKowhkaxUEqPUiQENeHmPA?e=DSynNh) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYtVl95nSypFvTJa8B5vSUIB9ibmgxwF9ACFNnDdjBWDXA?e=EvH5PS) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EbmGEmoL539HkBBPpC-SyagB4niZG9IlaNnF71mRuFqa7Q?e=9jHWY5) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EXskYXQgtFZOtu2t-FLzbC8BTqZr8QtRiqtcmOiWCZcpNg?e=05qYTo) |
| Segmentor-B | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EdDo45KYM9BPl3JGKsYaOZsB5lZOUzCZdy7jwZBxn4htGg?e=35cLRu) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EcgysPKrQP1Fs_oByRWAEngBuDjw3Kn6akZbtTl6Wj2hYg?e=nAb6za) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EfznT2AbW5VNiQIIfeq5h8sBvdoioMH35P9PA7bnF1igCQ) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EVvfF959JptOolf7xt8ZPdoBIGIM9UwpTCWDhSLVTtDV_w?e=bJuHZ5) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EQi4RBhTdvFGqIgNyj9UZ4QBTrmK9kJcLwJ4HMlPTHq53w?e=11r7IN) |
| Segmentor-L | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYLBmedg0nlAp5dqitn8pxcBo_9OcRWHOKpzb5Q9g5f8Kw?e=kie9IK) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/ET0D1YyinExLnum2L3y4soABiPgw_99AcocruqM4bw95pA?e=XVoDhq) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EWQ_o9jIIWVItezvJnpPmkEBQY38Agh0YGHlOHCQZGAIig?e=Foscbm) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EZ02oBK828dLo5P1Z1N9RV0BpzIum-8du7HXDCU4Ue8omg?e=Kt5v5r) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EWErh4qZWSxErgGGxZ_fQPQB3KXnGZ1iTJVtzwwn5sNJyg?e=8ZQo9m) |
| Segmentor-H | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYCdndKjn4NJv3Qvebo4YsQBrUhU_Uu2tjtBucJH2SMdNQ?e=NThF4d) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ebg9v0HaOFZIpyda-JKNST8B2AmnGdhgYQqjdLHYm4j5LA?e=ibANRv) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EZjmiotww1hHtF83WwJTVz0BAmfDNkuSuGbUXkthP3yvDQ?e=N45aU3) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EQG3IMH1OARPj67mapQoakYBjlkMzAKzQjYxPn425JiVeQ?e=6XrKmT) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EVj-vCQh5MVPqIT8ggkSJGsBeWv_MsrO9Ci3Lr7wuewW2A?e=qsa0Gd) |



