
from pydantic import BaseModel
from datetime import datetime, timezone
import pandas as pd

class LogRecord(BaseModel):
    """
        A pydantic basemodel to represent a log data header
    """

    log_id:int
    label:str
    timestamp:datetime
    node:str
    type:str
    component:str
    severity:str
    message:str


fields =[
    "label",
    "timestamp",
    "date",
    "node",
    "datetime",
    "node_2",
    "component",
    "category",
    "severity",
    "message"
]


# with open(r'E:\incidentiq\data\raw\BGL_2k.log') as file:
#     nodes = set()
#     severities = set()
#     for line in file:
#         data = line.split(maxsplit=9)
#         nodes.add(data[3])
#         severities.add(data[8])
#         data_dict = dict(zip(fields,data))
#         log = LogRecord(**data_dict)
#         print(log)
logs = []
with open(r"E:\incidentiq\data\raw\BGL_2k.log") as file:
    # nodes = set()
    # severities = set()
    # components = set()
    # categories = set()
    # severity_counts = {}
    # labels = {}
    # combined_freq = {}
    # type_counts = {}
    # label_counts = {}
    # component_counts = {}
    # label_comp = {}
    # for line in file:
        # data = line.split(maxsplit=9)
        # labels[data[0]] = labels.get(data[0], 0) + 1
        # print(data)
        # nodes.add(data[3])
        # nodes.add(data[5])
        # severities.add(data[8])
        # components.add(data[6])
        # categories.add(data[7])
        # severity_counts[data[8]] = severity_counts.get(data[8],0) + 1
        # print(data)
        # time = float(data[1])
        # print(datetime.fromtimestamp(time))
        # print(datetime.fromtimestamp(time, timezone.utc))
        # combined_freq[data[0]+" "+data[6]+" "+data[7]] = combined_freq.get(data[0]+data[6]+data[7], 0) + 1
        # type_counts[data[6]] = type_counts.get(data[6],0) + 1
        # component_counts[data[7]] = component_counts.get(data[7],0) + 1
        # label_counts[data[0]] = label_counts.get(data[0],0) + 1
        # components = label_comp.setdefault(data[0],set())
        # components.add(data[7])
    # line = file.readline().rstrip("\n")
    id = 1
    for line in file:
        data = line.split(maxsplit=9)
        timestamp = datetime.strptime(data[4],"%Y-%m-%d-%H.%M.%S.%f")
        log = LogRecord(
            log_id=id,
            label=data[0],
            timestamp=timestamp,
            node=data[3],
            type=data[6],
            component=data[7],
            severity=data[8],
            message=data[9].rstrip("\n")
            )
        id += 1
        logs.append(log)   




    # print(nodes)
    # print(severities)
    # print(components)
    # print(categories)
    # print(f"Nodes: {len(nodes)}, severities: {len(severities)}, components: {len(components)}")
    # print(labels)
    # print(severity_counts)
    # print(combined_freq)
    # print(type_counts)
    # print(component_counts)
    # print(label_counts)
    # print(label_comp)
    print(logs[0])
    print(logs[-1])

    records = [log.model_dump() for log in logs]
    df = pd.DataFrame(records)

    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)

    fatal_logs = df[df["severity"]=="FATAL"]
    print(len(fatal_logs))

    df.to_parquet(
        r"E:\incidentiq\data\processed\logs.parquet",
        index=False
    )