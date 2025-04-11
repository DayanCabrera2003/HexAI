import time
import math
from heapq import heappush, heappop
from collections import deque
from hex_board import HexBoard
from player import Player

class SmartAIPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 3 - player_id
        self.size = None
        self.opening_book = self.create_opening_book()
        self.direction_weights = None
        self.path_cache = {}
        self.zobrist_cache = {}
        self.last_best_path = []

    def create_opening_book(self):
        return {
            7: [(3,3), (2,2), (4,4), (3,4), (2,3)],
            11: [(5,5), (4,5), (6,5), (5,4), (5,6)],
            13: [(6,6), (5,6), (7,6), (6,5), (6,7)],
            16: [(8,8), (7,8), (9,8), (8,7), (8,9)]
        }

    def play(self, game: HexBoard, max_time: float) -> tuple:
        start_time = time.time()
        self.size = game.size
        self.direction_weights = self.precompute_direction_weights()
        self.path_cache.clear()
        
        # Paso 1: Verificar victoria inmediata
        if win_move := self.find_winning_move(game):
            return win_move
        
        # Paso 2: Usar apertura predefinida
        occupied = sum(1 for row in game.board for cell in row if cell != 0)
        if occupied < 2 and (opening_move := self.get_opening_move(game)):
            return opening_move
        
        # Paso 3: Búsqueda adaptativa con gestión de tiempo
        best_move = None
        depth = 3
        time_limit = start_time + max_time * 0.95
        
        while time.time() < time_limit and depth <= 15:
            try:
                current_move, _ = self.alpha_beta_search(game, depth, time_limit)
                if current_move:
                    best_move = current_move
                depth += 1 if depth < 7 else 2
            except TimeoutError:
                break
                
        # Paso 4: Fallback estratégico
        return best_move or self.strategic_fallback(game)

    def alpha_beta_search(self, board, depth, time_limit):
        def recurse(node, alpha, beta, depth, maximizing):
            if time.time() > time_limit:
                raise TimeoutError()

            # Verificar estado terminal
            if node.check_connection(self.player_id):
                return (None, math.inf)
            if node.check_connection(self.opponent_id):
                return (None, -math.inf)
            if not node.get_possible_moves():
                return (None, 0)

            if depth == 0:
                return (None, self.evaluate_position(node))

            best_value = -math.inf if maximizing else math.inf
            best_move = None
            moves = self.get_ordered_moves(node, maximizing)

            for move in moves:
                child = node.clone()
                child.place_piece(*move, self.player_id if maximizing else self.opponent_id)
                
                _, value = recurse(child, alpha, beta, depth-1, not maximizing)
                
                if maximizing:
                    if value > best_value:
                        best_value = value
                        best_move = move
                        alpha = max(alpha, value)
                else:
                    if value < best_value:
                        best_value = value
                        best_move = move
                        beta = min(beta, value)
                        
                if alpha >= beta:
                    break

            return (best_move, best_value)

        try:
            return recurse(board.clone(), -math.inf, math.inf, depth, True)
        except TimeoutError:
            return (None, 0)

    def calculate_blocking_potential(self, board):
        """Nuevo método para calcular potencial de bloqueo"""
        opponent_cost = self.calculate_path_cost(board, self.opponent_id)
        if opponent_cost == math.inf:
            return 0.0
        return 1.0 / (opponent_cost + 1)
    
    def evaluate_position(self, board):
        """Evaluación estratégica con prioridad en extensión de camino y bloqueo"""
        score = 0
        
        # 1. Evaluar puentes críticos
        critical_score = self.evaluate_critical_bridges(board)
        if abs(critical_score) > 0:
            return critical_score * 3.0
        
        # 2. Progreso en camino principal
        my_progress = self.calculate_path_progress(board, self.player_id)
        opp_progress = self.calculate_path_progress(board, self.opponent_id)
        score += (my_progress - opp_progress) * 2.8
        
        # 3. Bloqueo estratégico
        score += self.calculate_blocking_potential(board) * 1.5
        
        # 4. Control central dinámico
        score += self.dynamic_center_control(board) * 0.8
        
        return score

    def calculate_path_progress(self, board, player_id):
        """Calcula el progreso hacia el borde objetivo con ponderación direccional"""
        max_progress = 0
        starts = [(i, 0) for i in range(self.size)] if player_id == 1 else [(0, i) for i in range(self.size)]
        
        for start in starts:
            if board.board[start[0]][start[1]] != player_id:
                continue
                
            visited = set()
            queue = deque([(start[0], start[1], 0)])
            
            while queue:
                r, c, progress = queue.popleft()
                current_progress = c if player_id == 1 else r
                max_progress = max(max_progress, current_progress)
                
                for nr, nc in self.get_neighbors((r, c)):
                    if board.board[nr][nc] == player_id and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        new_progress = progress + (1 if (nc > c) else 0.2) if player_id == 1 else (
                                      progress + (1 if (nr > r) else 0.2))
                        queue.append((nr, nc, new_progress))
                        
        return max_progress

    def is_winning_move(self, move, board):
        """Verifica si un movimiento específico resulta en victoria inmediata"""
        temp_board = board.clone()
        temp_board.place_piece(*move, self.player_id)
        return temp_board.check_connection(self.player_id)
    
    def get_ordered_moves(self, board, maximizing):

        """Ordenamiento inteligente en 4 niveles de prioridad"""
        moves = board.get_possible_moves()
        
        # Nivel 1: Movimientos ganadores
        if wins := [m for m in moves if self.is_winning_move(m, board)]:
            return wins
            
        # Nivel 2: Bloqueo de amenazas críticas
        if threats := self.detect_critical_threats(board):
            return threats
            
        # Nivel 3: Extensión de camino propio
        return sorted(moves, key=lambda m: self.path_extension_score(m, board), reverse=maximizing)

    def path_extension_score(self, move, board):
        """Puntúa movimientos por progreso y bloqueo simultáneo"""
        temp_board = board.clone()
        temp_board.place_piece(*move, self.player_id)
        
        # Progreso propio
        progress_gain = self.calculate_path_progress(temp_board, self.player_id) - self.calculate_path_progress(board, self.player_id)
        
        # Bloqueo al oponente
        blocking_score = self.calculate_blocking_score(move, board)
        
        # Conexión con el camino principal
        connection_score = sum(1 for n in self.get_neighbors(move) if board.board[n[0]][n[1]] == self.player_id)
        
        return (progress_gain * 2.5) + (blocking_score * 1.8) + (connection_score * 1.2)

    def calculate_blocking_score(self, move, board):
        """Calcula cuánto bloquea el movimiento al oponente"""
        blocking = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
            nr, nc = move[0] + dr, move[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if board.board[nr][nc] == self.opponent_id:
                    blocking += 1
                # Penalizar movimientos cerca del camino del oponente
                if self.is_near_opponent_path((nr, nc), board):
                    blocking += 0.5
        return blocking

    def is_near_opponent_path(self, pos, board):
        """Verifica si la posición está cerca del camino principal del oponente"""
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = pos[0] + dr, pos[1] + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    if board.board[r][c] == self.opponent_id:
                        return True
        return False

    def strategic_fallback(self, board):
        """Estrategia de respaldo con enfoque en progreso y bloqueo"""
        moves = board.get_possible_moves()
        
        # Calcular puntuaciones para cada movimiento
        scored_moves = [(m, self.calculate_strategic_score(m, board)) for m in moves]
        
        # Seleccionar el movimiento con mayor puntuación
        best_move = max(scored_moves, key=lambda x: x[1], default=None)
        return best_move[0] if best_move else self.get_edge_progression_move(board)

    def calculate_strategic_score(self, move, board):
        """Puntúa movimientos basado en múltiples factores estratégicos"""
        score = 0
        
        # Progresión hacia el borde
        if self.player_id == 1:
            score += (move[1] / self.size) * 2.0  # Priorizar columnas derechas
        else:
            score += (move[0] / self.size) * 2.0  # Priorizar filas inferiores
            
        # Bloqueo al oponente
        score += self.calculate_blocking_score(move, board) * 1.5
        
        # Conexión con piezas existentes
        score += sum(1 for n in self.get_neighbors(move) if board.board[n[0]][n[1]] == self.player_id) * 1.2
        
        # Distancia al centro
        center = self.size // 2
        score -= (abs(move[0] - center) + abs(move[1] - center)) * 0.3
        
        return score

    def dynamic_center_control(self, board):
        """Control del centro con ponderación adaptativa"""
        center = self.size // 2
        control = 0
        for r in range(max(0, center-3), min(self.size, center+4)):
            for c in range(max(0, center-3), min(self.size, center+4)):
                distance = max(abs(r - center), abs(c - center))
                weight = 1.0 / (distance + 1)
                if board.board[r][c] == self.player_id:
                    control += weight
                elif board.board[r][c] == self.opponent_id:
                    control -= weight * 1.2  # Penalizar más la presencia del oponente
        return control

    def evaluate_critical_bridges(self, board):
        """Evalúa puentes con impacto directo en la victoria"""
        critical_score = 0
        
        # Puentes ofensivos
        for move in board.get_possible_moves():
            if self.is_winning_bridge(move, board):
                critical_score += 4.0
                
        # Puentes defensivos
        opponent_threats = self.detect_critical_threats(board)
        critical_score -= len(opponent_threats) * 3.5
        
        return critical_score

    def is_winning_bridge(self, move, board):
        """Verifica si el movimiento crea un puente ganador"""
        temp_board = board.clone()
        temp_board.place_piece(*move, self.player_id)
        
        # Verificar conexión directa
        if temp_board.check_connection(self.player_id):
            return True
            
        # Verificar puentes estratégicos en diagonal
        for dx, dy in [(-1, 1), (1, -1)]:
            cells = [
                (move[0] + dx, move[1] + dy),
                (move[0] - dx, move[1] - dy)
            ]
            if all(0 <= x < self.size and 0 <= y < self.size and
                   temp_board.board[x][y] == self.player_id for x, y in cells):
                return True
        return False

    def detect_critical_threats(self, board):
        """Detecta movimientos que permitirían al oponente ganar"""
        threats = []
        for move in board.get_possible_moves():
            temp_board = board.clone()
            temp_board.place_piece(*move, self.opponent_id)
            if temp_board.check_connection(self.opponent_id):
                threats.append(move)
        return threats

    def get_edge_progression_move(self, board):
        """Selecciona el movimiento que más avanza hacia el borde objetivo"""
        moves = board.get_possible_moves()
        if self.player_id == 1:
            # Priorizar progresión horizontal con equilibrio central
            return max(moves, key=lambda m: (m[1] * 2 - abs(m[0] - self.size//2)))
        else:
            # Priorizar progresión vertical con equilibrio central
            return max(moves, key=lambda m: (m[0] * 2 - abs(m[1] - self.size//2)))

    def find_winning_move(self, board):
        """Busca movimientos ganadores inmediatos"""
        for move in board.get_possible_moves():
            temp_board = board.clone()
            temp_board.place_piece(*move, self.player_id)
            if temp_board.check_connection(self.player_id):
                return move
        return None

    def get_opening_move(self, board):
        if self.size in self.opening_book:
            for move in self.opening_book[self.size]:
                if board.is_valid_move(*move):
                    return move
        return None

    def precompute_direction_weights(self):
        """Precalcula pesos para guiar la búsqueda de caminos"""
        weights = {}
        for r in range(self.size):
            for c in range(self.size):
                if self.player_id == 1:
                    # Priorizar progresión horizontal (columnas)
                    weights[(r, c)] = (self.size - c) * 0.8 + (self.size/2 - abs(r - self.size/2)) * 0.2
                else:
                    # Priorizar progresión vertical (filas)
                    weights[(r, c)] = (self.size - r) * 0.8 + (self.size/2 - abs(c - self.size/2)) * 0.2
        return weights

    def get_neighbors(self, pos):
        r, c = pos
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]
                if 0 <= r+dr < self.size and 0 <= c+dc < self.size]

    def calculate_path_cost(self, board, player_id):
        """A* optimizado con priorización direccional"""
        cache_key = (str(board.board), player_id)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        starts = [(i, 0) for i in range(self.size)] if player_id == 1 else [(0, i) for i in range(self.size)]
        heap = []
        visited = {}
        
        for pos in starts:
            cost = 0 if board.board[pos[0]][pos[1]] == player_id else 1
            heappush(heap, (cost + self.direction_weights[pos], pos))
            visited[pos] = cost

        while heap:
            _, pos = heappop(heap)
            current_cost = visited[pos]
            
            if (player_id == 1 and pos[1] == self.size-1) or (player_id == 2 and pos[0] == self.size-1):
                self.path_cache[cache_key] = current_cost
                return current_cost
                
            for neighbor in self.get_neighbors(pos):
                nr, nc = neighbor
                if board.board[nr][nc] == 3 - player_id:
                    continue
                    
                new_cost = current_cost + (0 if board.board[nr][nc] == player_id else 1)
                if neighbor not in visited or new_cost < visited.get(neighbor, math.inf):
                    visited[neighbor] = new_cost
                    priority = new_cost + self.direction_weights[neighbor]
                    heappush(heap, (priority, neighbor))
                    
        self.path_cache[cache_key] = math.inf
        return math.inf