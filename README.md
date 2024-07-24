# mobis_PLC

## Requirements
```
conda create -n mobis python=3.10
conda activate mobis
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

** 최신 모델(e.g. Llama-3.1) 을 vLLM에서 로드하기 위해서는 배포 패키지를 install **
https://github.com/vllm-project/vllm/releases

vLLM 내에서 Flash-Attention 사용하기 위해서 cuda 12.1 버전 install 추천
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
혹은 pip3 install torch torchvision torchaudio

```
