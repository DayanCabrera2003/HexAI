# IA Avanzada para Hex 🎮🤖

IA inteligente para el juego Hex que combina búsqueda adaptativa con estrategias posicionales avanzadas, con sistema de bloqueo dinámico y gestión de tiempo óptima.

## Métodos Clave 🔑

### `create_opening_book()`
- **Descripción:** Genera aperturas predefinidas para diferentes tamaños de tablero
- **Recibe:** Nada
- **Devuelve:** Diccionario con aperturas para tamaños 7, 11, 13 y 16
- **Estrategia:** Control central inicial + conexiones diagonales

### `play(game: HexBoard, max_time: float)`
- **Descripción:** Método principal que ejecuta la IA
- **Recibe:** 
  - `game`: Estado actual del tablero
  - `max_time`: Tiempo máximo permitido
- **Devuelve:** Tupla (fila, columna) con mejor movimiento
- **Flujo:**
  1. Verificación de victoria inmediata
  2. Uso de aperturas predefinidas
  3. Búsqueda alfa-beta adaptativa
  4. Fallback estratégico

### `alpha_beta_search(board, depth, time_limit)`
- **Descripción:** Búsqueda con poda alfa-beta optimizada
- **Recibe:**
  - `board`: Tablero actual
  - `depth`: Profundidad inicial
  - `time_limit`: Límite temporal
- **Devuelve:** Mejor movimiento y valor asociado
- **Innovación:** Gestión dinámica de tiempo + ordenamiento inteligente de movimientos

### `evaluate_position(board)`
- **Descripción:** Evaluación heurística multifactorial
- **Factores:**
  1. Puentes críticos (30%)
  2. Progreso en camino principal (25%)
  3. Bloqueo estratégico (20%)
  4. Control central (15%)
  5. Conexiones potenciales (10%)
- **Métrica:** Puntuación normalizada [-100, 100]
## Técnicas Clave 🔥

### 1. Evaluación Heurística Multidimensional  
**Componentes:**  
| Factor               | Peso  | Descripción                              |
|----------------------|-------|------------------------------------------|
| Progreso del Camino  | 30%   | Avance hacia el borde objetivo           |
| Bloqueo Estratégico  | 25%   | Interrupción de rutas rivales            |
| Puentes Críticos     | 20%   | Conexiones diagonales decisivas          |
| Control Central      | 15%   | Dominio del área núcleo                  |
| Movilidad            | 10%   | Flexibilidad para futuras jugadas        |

puntuación = sum(
    progreso * 0.3,
    bloqueo * 0.25,
    puentes * 0.2,
    control * 0.15,
    movilidad * 0.1
)
### 2. Gestión Inteligente de Tiempo ⏳
def ajustar_profundidad(tiempo_restante: float) -> int:
    if tiempo_restante > 5.0:
        return 3  # Máxima profundidad
    elif tiempo_restante > 2.5:
        return 2  # Profundidad media
    else:
        return 1  # Búsqueda rápida
        
## Estrategia Global 🧠

### 1. Sistema de Prioridades Adaptativo
- **Nivel 1:** Movimientos ganadores inmediatos
- **Nivel 2:** Bloqueo de amenazas críticas
- **Nivel 3:** Extensión de camino propio
- **Nivel 4:** Desarrollo posicional

### 2. Mecanismo de Búsqueda
```mermaid
graph TD
    A[Inicio] --> B{¿Victoria inmediata?}
    B -->|Sí| C[Jugar movimiento]
    B -->|No| D{¿Apertura?}
    D -->|Sí| E[Jugar apertura]
    D -->|No| F[Búsqueda Alfa-Beta]
    F --> G[Ordenar movimientos]
    G --> H[Evaluar subárboles]
    H --> I{¿Mejor que actual?}
    I -->|Sí| J[Actualizar mejor]
    I -->|No| K[Podar rama]

