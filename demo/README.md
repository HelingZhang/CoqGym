Given an arbitrary proof in `scratch.v`, extract information including proof tree into a json file with the following procedure:
0.  Make sure to have their modified version of coq installed. To install, go to `CoqGym/` and run `source install.sh`. (for detail refer to `CoqGym/README.md)
1.  go to `CoqGym/demo`, run `source coqproject.sh`
2.  run `make`. Now there should be a `scratch.meta` file under current directory (`CoqGym/demo/`) 
3.  run `bash extract_json.sh`

Now there should be a `scratch.json` under `CoqGym/data/demo`. This file contains information about every proof in `scratch.v`. We can use `scratch.json` to interact with CoqGym's proof evaluation api by running `python eval_env.py --file data/demo/scratch.json` under `CoqGym`.