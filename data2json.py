import jsonlines
with open('/home/ypd-19-2/prefix-tuning/ypd_prefix/multiwoz_transfer_dataset/train_as_test/test.source', 'r') as r:
    train_source = r.readlines()
with open('/home/ypd-19-2/prefix-tuning/ypd_prefix/multiwoz_transfer_dataset/train_as_test/test.target', 'r') as r:
    train_target = r.readlines()
w = jsonlines.open('/home/ypd-19-2/docomo/data_source_target/train/test.json', 'w')
for s, t in zip(train_source, train_target):
    line_dict = {}
    line_dict["text"] = s
    line_dict["summary"] = t
    jsonlines.Writer.write(w, line_dict)
