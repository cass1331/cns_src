import csv


class CSVLogger():
    def __init__(self,  args,fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def final(self,best_accuracy_list,final_accuracy_list,best_epoch_list,ave_valid_acc_50_list,acc_each_class_list,acc_each_dataset_list = None):
        writer = csv.writer(self.csv_file)
        best_accuracy_list.insert(0,'best acc')
        final_accuracy_list.insert(0,'final acc')
        best_epoch_list.insert(0,'best epoch')
        ave_valid_acc_50_list.insert(0,'ave_valid_acc_50')
        acc_each_class_list.insert(0,'acc_each_class')
        if acc_each_dataset_list!= None:
            acc_each_dataset_list.insert(0,'acc_each_dataset')
            writer.writerow(acc_each_dataset_list)
        writer.writerow(best_accuracy_list)
        writer.writerow(final_accuracy_list)
        writer.writerow(best_epoch_list)
        writer.writerow(ave_valid_acc_50_list)
        writer.writerow(acc_each_class_list)
        
        self.csv_file.flush()
    def close(self):
        self.csv_file.close()