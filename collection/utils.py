import csv
import os


def write_csv_data(csv_path, data_iterator, target_attrs):
    """
    Opens file from csv_path, and receives data from data_iterator,
    extracting attributes in target_attrs from json and writing to .csv
    """

    if os.path.exists(csv_path):
        option = input('''Path %s already exists. Select one of the following options:
        1. Append to %s
        2. Delete %s and write new
        3. Exit application
        ''' % (csv_path, csv_path, csv_path))

        if option == '1':
            pass
        elif option == '2':
            os.remove(csv_path)
        else:
            exit(0)
    else:
        directory = os.path.dirname(csv_path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

    f = open(csv_path, 'w+')
    w = csv.writer(f, delimiter=',')

    w.writerow(target_attrs)

    num_written = 0
    ids = set()

    print('Staged to write data to %s' % csv_path)

    for d in data_iterator:
        d['id'] = int(d['id'])  # TODO: alter data so all ids are already int

        if d['id'] in ids:
            continue  # if already seen id

        ids.add(d['id'])

        save_data = [value for (key, value) in d.items() if key in target_attrs]

        w.writerow(save_data)
        num_written += 1

    print('Wrote %i rows to %s' % (num_written, csv_path))

    f.close()
    print('Done\n')
