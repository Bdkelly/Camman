import json

def creator(ent):
    pass
    

def breaker(jsond):
    videoname = ""
    for i in jsond:
        for k,v in i.items():
            if k == "videos":
                for kn,vn in v.items():
                    videoname = kn
        print(i)
        print(videoname)
        
                



if __name__ == "__main__":
    with open("my_combined_output.json",'r') as f:
        dj = json.load(f)
    breaker(dj)
    