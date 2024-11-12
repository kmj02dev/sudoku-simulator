from typing import List
import reflex as rx
import asyncio
import time
import sys
sys.setrecursionlimit(1000000)
# ----------------------------------------------------- Backend Area --------------------------------------------------
class Board:
    def __init__(self, board = [[0 for _ in range(9)] for _ in range(9)], 
                 selected_cell = (0,0), 
                 count = 0, 
                 start_time = 0.0, 
                 end_time = 0.0, 
                 dirty = False):
        self.board: List[List[int]] = board
        self.selected_cell: tuple = selected_cell
        self.count: int = count
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.dirty: bool = dirty
    
    def get(self, row, col):
        return self.board[row][col]
    
    def get_board(self):
        return self.board
    
    def copy_board(self): # deep copy
        return [row[:] for row in self.board]
    
    def copy_myself(self): # deep copy
        # copied_board = Board(self.copy_board(), self.selected_cell, self.count, self.start_time, self.end_time, self.dirty) # board is shallowly copied
        # return copied_board
        pass
    
    def insert_cell(self, row: int, col: int, number: int):
        if self.board[row][col] == 0:
            self.board[row][col] = number
        
        if not self.dirty:
            self.dirty = True
            self.start_time = time.time()

        self.end_time = time.time()
        self.selected_cell = (row, col)
        self.count += 1

    def delete_cell(self, row: int, col: int):
        self.board[row][col] = 0 # DANGER !!!!!!!!!!!!!!!!!!!!!!!!!!

        if not self.dirty:
            self.dirty = True
            self.start_time = time.time()
            
        self.end_time = time.time()
        self.selected_cell = (row, col)
        self.count += 1

    def is_valid_move(self, row: int, col: int, number: int) -> bool:
        # check row
        if number in self.board[row]:
            return False

        # check column
        if number in [self.board[i][col] for i in range(9)]:
            return False

        # check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.board[i][j] == number:
                    return False

        return True
    
class BoardHistory:
    def __init__(self, original_board : Board):
        self.board_history : List[Board] = [original_board]

    def get_original_board(self):
        return self.board_history[0]
    
    def get_history(self):
        return self.board_history
    
    def get_board(self, time:int):
        return self.board_history[time]
    
    def append(self, board: Board):
        self.board_history.append(board)

class Simulator:
    def __init__(self, base_puzzle):
        base_puzzle = base_puzzle
        base_board = Board(board = base_puzzle)

        self.board_history : BoardHistory = BoardHistory(base_board)
        self.now_board : Board = base_board

    def insert_cell(self, row : int, col : int, num : int):
        self._swap()
        self.now_board.insert_cell(row, col, num)

    def delete_cell(self, row : int, col : int):
        self._swap()
        self.now_board.delete_cell(row, col)
    
    def get_consequence(self):
        return self.board_history.get_history() + [self.now_board]
    
    def _swap(self):
        self.board_history.append(self.now_board)
        copied_board = Board(self.now_board.copy_board(), 
                             self.now_board.selected_cell, 
                             self.now_board.count, 
                             self.now_board.start_time, 
                             self.now_board.end_time, 
                             self.now_board.dirty)
        self.now_board = copied_board

class SudokuSolver:
    def __init__(self, base_puzzle):
        self.simulator = Simulator
    
    def solve(self):
        pass

    def get_history(self) -> List[Board]:
        pass

class FakeSolver:
    def __init__(self, base_puzzle):
        self.simulator = Simulator(base_puzzle=base_puzzle)
    
    def solve(self):
        for i in range(9):
            for j in range(9):
                self.simulator.insert_cell(i, j, 1)
    
    def get_history(self):
        return self.simulator.get_consequence()

class ConstraintPropagationSolver:
    def __init__(self, base_puzzle):
        self.simulator = Simulator(base_puzzle=base_puzzle)
        
    def solve(self):
        updated = False  # a flag to track if any updates
        for row in range(9):
            for col in range(9):
                if self.simulator.now_board.board[row][col] == 0:
                    candidates = self.find_candidates(row, col) 
                    if len(candidates) == 1:
                        # If there is exactly one candidate, fill the cell
                        num = candidates.pop()
                        self.simulator.insert_cell(row, col, num)
                        updated = True
        if updated:
            self.solve()  # Continue until no more updates  
        else:
            self.backtracking()

    def find_candidates(self, row: int, col: int) -> set[int]:
        candidates = set(range(1, 10))
        # possible numbers that can be placed in the empty cell.
        candidates -= set(self.simulator.now_board.board[row])  # Remove numbers already present in the same row
        candidates -= {self.simulator.now_board.board[i][col] for i in range(9)}  # Remove numbers already present in the same column
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):  # Remove numbers already present in the 3x3 subgrid containing the cell
            for j in range(box_col, box_col + 3):
                candidates.discard(self.simulator.now_board.board[i][j])
        return candidates

    def backtracking(self):
        def backtrack(row: int, col: int) -> bool:
            # If we reach the end of the board, return True
            if row == 9:
                return True
            
            # If this cell is already filled, move to the next cell  
            if self.simulator.now_board.board[row][col] != 0:
                return backtrack(row + (col + 1) // 9, (col + 1) % 9)
            
            # Get the possible candidates for this cell
            candidates = self.find_candidates(row, col)
            
            for num in candidates:
                if self.simulator.now_board.is_valid_move(row, col, num):
                    self.simulator.insert_cell(row, col, num)
                    
                    # Recursively attempt to solve from the next cell
                    if backtrack(row + (col + 1) // 9, (col + 1) % 9):
                        return True
                    
                    # If no solution is found, backtrack and try next candidate
                    self.simulator.delete_cell(row, col)
            
            return False  # No valid solution found for this cell
    
    # Start backtracking from the top-left corner (0,0)
        return backtrack(0, 0)
    
    def get_history(self):
        return self.simulator.get_consequence()
    
class BackTrackingSolver:
    def __init__(self, base_puzzle):
        self.simulator = Simulator(base_puzzle=base_puzzle)

    def solve(self):
        self.backtracking()

    def backtracking(self):
        def backtrack(row: int, col: int) -> bool:
            # If we reach the end of the board, return True
            if row == 9:
                return True
            
            # If this cell is already filled, move to the next cell  
            if self.simulator.now_board.board[row][col] != 0:
                return backtrack(row + (col + 1) // 9, (col + 1) % 9)
            
            for num in range(1, 10):
                if self.simulator.now_board.is_valid_move(row, col, num):
                    self.simulator.insert_cell(row, col, num)
                    
                    # Recursively attempt to solve from the next cell
                    if backtrack(row + (col + 1) // 9, (col + 1) % 9):
                        return True
                    
                    # If no solution is found, backtrack and try next candidate
                    self.simulator.delete_cell(row, col)
            
            return False  # No valid solution found for this cell
    
        # Start backtracking from the top-left corner (0,0)
        backtrack(0, 0)

    def get_history(self):
        return self.simulator.get_consequence()

class BitmaskSudokuSolver:
    def __init__(self, base_puzzle):
        self.simulator = Simulator(base_puzzle=base_puzzle)

    def get_history(self):
        return self.simulator.get_consequence()
    
    def solve(self):
        rows = [0] * 9        # bitmask for rows
        cols = [0] * 9        # bitmask for cols
        boxes = [0] * 9       # bitmask for boxes

        # set exist number to bitmask
        for i in range(9):
            for j in range(9):
                num = self.simulator.now_board.board[i][j]
                if num != 0:
                    bit = 1 << (num - 1)
                    rows[i] |= bit
                    cols[j] |= bit
                    boxes[(i // 3) * 3 + j // 3] |= bit

    
        def backtrack(row=0, col=0):
            if row == 9:  
                return True
            if col == 9:  
                return backtrack(row + 1, 0)
            if self.simulator.now_board.board[row][col] != 0:  
                return backtrack(row, col + 1)

            for num in range(1, 10):
                bit = 1 << (num - 1)
                box_index = (row // 3) * 3 + col // 3
                if not (rows[row] & bit or cols[col] & bit or boxes[box_index] & bit):
                    self.simulator.insert_cell(row, col, num)
                    rows[row] |= bit
                    cols[col] |= bit
                    boxes[box_index] |= bit

                    # move next cell
                    if backtrack(row, col + 1):
                        return True

                    # undo
                    self.simulator.delete_cell(row, col)
                    rows[row] &= ~bit
                    cols[col] &= ~bit
                    boxes[box_index] &= ~bit

            return False  

        backtrack()

class Solution:
    ConstraintPropagation = "Constratint Propatation"
    BackTracking = "Back Tracking"
    Bitmask = "Bitmask"

class Level:
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class TestCase:
    easy_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    
    hard_puzzle = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ]
    
class State(rx.State):
    board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    selected_cell: tuple = (0, 0)
    original_board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]

    count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    strategy: str = Solution.BackTracking
    level: str = Level.EASY

    is_solved: bool = False
    is_running: bool = False
    is_fast_mode: bool = False
    delay: float = 0.0

    def init_board(self):
        base_puzzle = None
        if self.level == Level.EASY:
            base_puzzle = [row[:] for row in TestCase.easy_puzzle]
        elif self.level == Level.HARD:
            base_puzzle = [row[:] for row in TestCase.hard_puzzle]

        self.board = [row[:] for row in base_puzzle]
        self.original_board = [row[:] for row in base_puzzle]
        self.count = 0
        self.start_time = 0
        self.end_time = 0
        self.is_solved = 0

    async def solve(self):
        self.is_running = True

        base_puzzle = None
        if self.level == Level.EASY:
            base_puzzle = [row[:] for row in TestCase.easy_puzzle]
        elif self.level == Level.HARD:
            base_puzzle = [row[:] for row in TestCase.hard_puzzle]

        solver = None
        if self.strategy == Solution.ConstraintPropagation:
            solver = ConstraintPropagationSolver(base_puzzle)
        elif self.strategy == Solution.BackTracking:
            solver = BackTrackingSolver(base_puzzle)
        elif self.strategy == Solution.Bitmask:
            solver = BitmaskSudokuSolver(base_puzzle)

        self.is_solved = False
        self.start_time = time.time()
        solver.solve()
        self.end_time = time.time()
        self.is_solved = True

        board_history = solver.get_history()

        for board in board_history:
            self.board = board.board
            self.selected_cell = board.selected_cell
            self.count = board.count
            # self.start_time = board.start_time
            # self.end_time = board.end_time
            if not self.is_fast_mode : 
                await asyncio.sleep(self.delay)
                yield
        self.is_running = False

# ----------------------------------------------------- Frontend Area --------------------------------------------------
def box(item, row, col):
  return rx.box(
      item,
      border="1px solid #CBD5E0",
      width="3em",
      height="3em",
      display="flex",
      align_items="center",
      justify_content="center",
      font_size="1.2em",
      font_weight="bold",
      bg=rx.cond(
          row == State.selected_cell[0],
          rx.cond(col == State.selected_cell[1], 
              rx.cond(State.is_running, "#dfdff7", rx.cond((row // 3 + col // 3) % 2 == 1, "#f5f9fc", "white")),  # 선택된 셀 배경색
              rx.cond((row // 3 + col // 3) % 2 == 1, "#f5f9fc", "white")
          ),
          rx.cond((row // 3 + col // 3) % 2 == 1, "#f5f9fc", "white")
      ),
      color="#2D3748",  # 글자색은 항상 동일
      _hover={"bg": "#EDF2F7"},
  )

def row(item, i):
    return rx.hstack(
        rx.foreach(item, lambda item, j: box(item, i, j)),
        spacing="0"
    )

def upper_control_panel():
    return rx.hstack(
        rx.box(
            rx.hstack(
                rx.text("Processing time:", color="gray.600", font_size="sm"),
                rx.text(
                    f"{round((State.end_time - State.start_time)*10000) / 10000} sec", 
                    font_size="lg", 
                    font_weight="bold"
                ),
                spacing="1",
            ),
        ),
        rx.box(
            rx.hstack(
                rx.text("Insert/Delete Count:", color="gray.600", font_size="sm"),
                rx.text(f"{State.count}", font_size="lg", font_weight="bold"),
                spacing="1",
            ),
        ),
        rx.cond(State.is_solved, 
            rx.text("Solved", margin_left='auto', weight="bold", color="green"), 
            rx.text("Not Solved", margin_left='auto', weight="bold", color="red"),
        ),
        width="100%",
        justify="start",
        spacing="4",  # 요소 간 간격 추가
    )

def below_control_panel():
    return rx.hstack(
        rx.vstack(
            rx.hstack(
                rx.input(
                    placeholder="Delay",
                    on_blur=State.set_delay,
                    width="100px"
                ),
                rx.badge(f"x{State.delay}"),
                width="fit-content",
            ),
            rx.center(
                rx.badge("Instant Complete"),
                rx.switch(on_change=State.set_is_fast_mode),
                spacing="2"
            ),
            spacing="4",
            width="fit-content",
            align="start",
        ),
        rx.hstack(
            rx.vstack(
                rx.select(
                    [Level.EASY, Level.HARD],
                    value=State.level,
                    on_change=State.set_level,
                    width="150px",
                    border_radius="md",
                    border="1px solid #E2E8F0",
                ),
                rx.select(
                    [Solution.BackTracking, Solution.Bitmask, Solution.ConstraintPropagation],
                    value=State.strategy,
                    on_change=State.set_strategy,
                    width="150px",
                    border_radius="md",
                    border="1px solid #E2E8F0",
                ),
                
            ),
            rx.vstack(
                rx.button(
                    "Create Board",
                    on_click=State.init_board,
                    bg="blue.500",
                    color="white",
                    px="6",
                    py="2",
                    
                    border_radius="md",
                    _hover={"bg": "blue.600"},
                    disabled=State.is_running,  # 상태값에 따라 비활성화
                ),
                rx.button(
                    "Run",
                    on_click=State.solve,
                    bg="green.500",
                    color="white",
                    px="6",
                    py="2",
                    border_radius="md",
                    _hover={"bg": "green.600"},
                    disabled=State.is_running,  # 상태값에 따라 비활성화
                ),
                width="fit-content",
                spacing="3",
            ),
            width="fit-content",
            spacing="4",
        ),
        width="100%",
        justify="between"
        
    )

def index():
    return rx.center(
        rx.box(  # container를 box로 변경
            rx.vstack(
                rx.heading(
                    "Sudoku Simulator", 
                    size="xl", 
                    mb="4",
                    align_self="start",
                    color="gray.800",
                ),
                upper_control_panel(),
                rx.box(
                    rx.vstack(
                        rx.foreach(State.board, lambda item, i: row(item, i)),
                        align_items="center",
                        spacing="0",
                    ),
                    border="2px solid #2D3748",
                    border_radius="lg",
                    p="4",
                    bg="white",
                    shadow="lg",
                    my="6",  # 상하 마진 추가
                ),
                below_control_panel(),
                width="fit-content",  # 내용물 크기에 맞춤
                spacing="4",
                p="6",  # 내부 패딩
            ),
            width="fit-content",  # 내용물 크기에 맞춤
            mx="auto",  # 가운데 정렬
        ),
        width="100%",
        min_height="100vh",
    )

# ----------------------------------------------------- Page Area --------------------------------------------------
app = rx.App()
app.add_page(index)


