import csv
import os


def write_csv_data(csv_path, data_iterator, target_attrs):
    """
    Opens file from csv_path, and receives data_master from data_iterator,
    extracting attributes in target_attrs from json and writing to .csv
    """

    if os.path.exists(csv_path):
        overwrite = input('Path %s already exists. Do you want to overwrite it? Y/n ' % csv_path)

        if overwrite == 'Y':
            os.remove(csv_path)
        else:
            exit(0)

    directory = os.path.dirname(csv_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    f = open(csv_path, 'w+')
    w = csv.writer(f, delimiter=',')

    w.writerow(target_attrs)

    num_written = 0
    ids = set()

    print('Staged to write data_master to %s' % csv_path)

    for d in data_iterator:
        d['id'] = int(d['id'])  # TODO: alter data_master so all ids are already int

        if d['id'] in ids:
            continue  # if already seen id

        ids.add(d['id'])

        save_data = [value for (key, value) in d.items() if key in target_attrs]

        if len(save_data) == 1:  # if from connections TODO: clean up logic
            save_data = save_data[0]

        w.writerow(save_data)
        num_written += 1

    print('Wrote %i rows to %s' % (num_written, csv_path))

    f.close()
    print('Done\n')
