import csv


class AbstractAgent:

    def __init__(self):
        pass

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data, v, transactions):
        pass

    def setGameParams(self, gameParams, visualTraits, _forcedRandom):
        pass

    def record_final_results(self, player_idx, round_num, recieved, popularities, influence, extra_data, v,
                             transactions):
        pass

    # def write_generator_usage(self, generator_file, round_num) -> None:
    #     assert self.generator_to_use_idx is not None
    #     assert self.whoami is not None
    #     dir = '/'.join(generator_file.split('/')[:-1])
    #     file = generator_file.split('/')[-1]
    #     file_adj = f'{self.whoami}_{file}'
    #     file_clean = f'{dir}/{file_adj}'
    #
    #     with open(f'{file_clean}.csv', 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([round_num, self.generator_to_use_idx])
    def write_generator_usage(self, generator_file, round_num) -> None:
        assert self.generator_to_use_idx is not None

        with open(generator_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round_num, self.generator_to_use_idx])
