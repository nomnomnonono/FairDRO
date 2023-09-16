# FairDRO: Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization
In progress...

非公式のFairDRO (https://openreview.net/forum?id=Q-WfHzmiG9m) の再現実装です．

## 実行環境
docker-composeを用います．インストールしていない人は公式のドキュメントに従ってインストールしてください．

- Dockerイメージの作成
```bash
$ make build
```
- Dockerコンテナ起動
```bash
$ make up
```
- Dockerコンテナに入る
```bash
$ make exec
```
## 実行例
```bash
$ python main.py --rho $rho --lr 0.001 --wd 0.001 -s $seed -b 256 -d celeba --target Smiling --sens Male --arch $model -e 50 --root data/CelebA/
```
