import json, collections
for tag in ("native","sage"):
    for shp in ("64","128"):
        pass
import glob
for p in ["/tmp/grid_native.json","/tmp/grid_sage.json"]:
    ev=json.load(open(p))["traceEvents"]
    cats=collections.Counter(e.get("cat") for e in ev if e.get("ph")=="X")
    print("###",p,len(ev),dict(cats))
    for e in ev:
        if e.get("ph")=="X" and e.get("cat") not in ("cpu_op","user_annotation","python_function",None):
            print("   ", e.get("cat"), repr(e.get("name"))[:60], e.get("dur"), json.dumps(e.get("args",{}))[:300])
