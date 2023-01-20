import numpy as np
class phone_maps():
    def __init__(self, mapping_file):
        self.symbols_int_map = {}
        self.reduced_symbols_int_map = []
        self.int_symbols_map = {}
        self.reduced_int_symbols = []
        with open(mapping_file, "r") as file:
            mappings = file.readlines()
        self.num_symbols = len(mappings)
        self.num_reduced_sets = len(mappings[0].strip().split())
        self.mappings = np.chararray((self.num_symbols+1, self.num_reduced_sets), unicode=True, itemsize=5)
        self.mappings[:] = " "
        self.mappings[0] = [" ", " ", " "] 
        self.stripped_mappings = [ele.strip().split("\t") for ele in mappings]
        for i in range(1, len(self.mappings)):
            for j in range(len(self.stripped_mappings[i-1])):
                self.mappings[i][j] = str(self.stripped_mappings[i-1][j])

        self.symbols = sorted(list(set(self.mappings[:,0])))
        self.reduced_symbols = [sorted(list(set(self.mappings[:,i]))) for i in range(self.mappings.shape[-1])]        
        self.__get_symbol_int_map()
        self.__get_int_symbol_map()

    def get_reduced_set(self, ith_reduced):
        return self.reduced_symbols[ith_reduced]
    def __get_int_symbol_map(self):
        self.int_symbols_map = {v: k for k, v in self.symbols_int_map.items()}
        self.reduced_int_symbols = [{v: k for k, v in d.items()} for d in self.reduced_symbols_int_map]
    def __get_symbol_int_map(self):
        self.symbols_int_map = {}
        self.reduced_symbols_int_map = []
        for index,symbol in enumerate(self.symbols):
            self.symbols_int_map[str(symbol)] = index

        for reduced_set in self.reduced_symbols:
            temp_int_map = {}
            for index,symbol in enumerate(reduced_set):
                temp_int_map[str(symbol)] = index
            self.reduced_symbols_int_map.append(temp_int_map)
        return self
    def map_symbol_int(self, symbol, level=-1):
        return self.reduced_symbols_int_map[level][str(symbol)]
    def map_symbols_ints(self, symbols, level=-1):
        return [self.map_symbol_int(symbol) for symbol in symbols]
    def map_symbol_reduced(self, symbol, level=-1):
        found_index = np.where(self.mappings[:,0] == str(symbol))[0][0]
        return self.mappings[found_index][level]
    def map_symbols_reduced(self, symbols, level=-1):
        return [self.map_symbol_reduced(symbol, level=level) for symbol in symbols]
    def print_symbol_maps(self, level=-1):
        for row in self.mappings:
            print(row[0], row[level])
    def print_symbol_int_maps(self):
        for key in self.symbols_int_map:
            print(key, self.symbols_int_map[key])
    def print_reduced_symbol_int_maps(self, level=-1):
        for key in self.reduced_symbols_int_map[level]:
            print(key, self.reduced_symbols_int_map[level][key])
    def print_int_symbol_maps(self):
        for key in self.int_symbols_map:
            print(key, self.int_symbols_map[key])
    def print_reduced_int_symbol_maps(self, level=-1):
        for key in self.reduced_int_symbols[level]:
            print(key, self.reduced_int_symbols[level][key])
