Given an arbitrary proof in `scratch.v`, extract information including proof tree into a json file with the following procedure:
1.  run `source coqproject.sh`
2.  run `make`
3.  run `bash extract_json.sh`

Now there should be a `scratch.json` under `CoqGym/data/demo`. This file contains information about every proof in `scratch.v`. We can use `scratch.json` to interact with CoqGym's proof evaluation api by running `python eval_env.py -file data/demo/scratch.json`.