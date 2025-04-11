class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: 'HexBoard', max_time: float) -> tuple:
        raise NotImplementedError("¡Implementa este método!")

from collections import deque

class HexBoard:
    def __init__(self, size=7):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.last_player = None
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

    def clone(self):
        new_board = HexBoard(self.size)
        new_board.board = [row.copy() for row in self.board]
        new_board.last_player = self.last_player
        return new_board

    def is_valid_move(self, row, col):
        """Verifica si un movimiento es válido"""
        return (0 <= row < self.size and 
                0 <= col < self.size and 
                self.board[row][col] == 0)

    def place_piece(self, row, col, player):
        """Coloca una ficha en el tablero"""
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            self.last_player = player
            return True
        return False

    def check_connection(self, player):
        """Verifica si el jugador tiene una conexión ganadora"""
        visited = set()
        queue = deque()

        # Configurar puntos de inicio según el jugador
        if player == 1:
            start_nodes = [(i, 0) for i in range(self.size)]
            target_col = self.size - 1
        else:
            start_nodes = [(0, i) for i in range(self.size)]
            target_row = self.size - 1

        # Inicializar BFS
        for node in start_nodes:
            if self.board[node[0]][node[1]] == player:
                queue.append(node)
                visited.add(node)

        while queue:
            r, c = queue.popleft()

            # Verificar condición de victoria
            if (player == 1 and c == target_col) or (player == 2 and r == target_row):
                return True

            # Explorar vecinos
            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 
                    0 <= nc < self.size and 
                    self.board[nr][nc] == player and 
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def get_possible_moves(self):
        """Devuelve todos los movimientos posibles"""
        return [(r, c) for r in range(self.size) 
                      for c in range(self.size) 
                      if self.board[r][c] == 0]

    def print_board(self):
        """Imprime el tablero en formato ASCII"""
        for r in range(self.size):
            print(' ' * r, end='')
            for c in range(self.size):
                cell = self.board[r][c]
                print('X' if cell == 1 else 'O' if cell == 2 else '.', end=' ')
            print()