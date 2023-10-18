Dataset **SISVSE** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/s/B/CH/hFs7ahEb7Dq6mNhGG7iOSsKC88w8n098LZZM6tgQVDHIGIExqirVKiGHOtNZEec5B5DOLYLq8OK072qxsKJshY7IPCJfmPaffUMaAPv4rLLUvwvTcrEZyQ1fp3d6.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='SISVSE', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

