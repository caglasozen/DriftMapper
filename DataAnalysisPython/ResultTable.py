import csv


class ResultTable:
    def __init__(self, file_path):
        file = open(file_path, newline='')
        file_reader = csv.DictReader(file, delimiter=',')
        self.data = []
        self.column_names = file_reader.fieldnames
        attribute_column = []
        for row in file_reader:
            self.data.append(row)
            attribute_column.append(row['attributes'])
        self.attributes = self.extract_attributes(attribute_column)
        self.distance_matrix = self.generate_distance_matrix(self.data, self.attributes, self.column_names)

    @staticmethod
    def extract_attributes(attribute_column):
        attributes = []
        for row in attribute_column:
            row_attributes = row.split("_")
            for att in row_attributes:
                if att != "" and att not in attributes:
                    attributes.append(att)
        return attributes

    @staticmethod
    def generate_distance_matrix(data, attributes, column_names):
        distance_matrix = [[0 for _ in attributes] for _ in attributes]
        for row in data:
            row_distance = float(row[column_names[0]])
            row_attributes = row[column_names[-1]].split("_")
            for att1 in row_attributes:
                for att2 in row_attributes:
                    try:
                        i = attributes.index(att1)
                        j = attributes.index(att2)
                        distance_matrix[i][j] += row_distance
                    except ValueError:
                        print(att1 + "," + att2)
        return distance_matrix
