# FairDRO: Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization
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
## 実行方法
```bash
$ python main.py -c config/celeba.yaml
```
