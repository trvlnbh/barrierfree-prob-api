## Barrier-free probability API



* 이미지 파일을 바이너리 포맷으로 read한 형식 사용



##### Run server

* 단일 모델로 실행

```shell
$python barrierfree.py --mode=1
```

* Average ensemble로 실행

```shell
$python barrierfree.py --mode=2
```



##### JSON Request 본문:

```json
{
    "image": "Binary 포맷의 이미지"
}
```



##### Response:

```json
{
    "result": [
        ["Accessible", 0.0163721200078258],
        ["Inaccessible", 0.9836279153823853]
    ]
}
```

