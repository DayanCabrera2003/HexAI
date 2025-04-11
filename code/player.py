from hex_board import HexBoard
class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: 'HexBoard', max_time: float) -> tuple:
        raise NotImplementedError("¡Implementa este método!")