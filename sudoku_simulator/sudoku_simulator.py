from typing import List, Optional
import reflex as rx
import asyncio
import time
from enum import Enum
import threading
import math

# ----------------------------------------------------- Backend Area --------------------------------------------------
class Solution:
    FAKE = "Fake"
    ConstraintPropagation = "Constratint Propatation"
    BackTracking = "Back Tracking"

class State(rx.State):
    # sudoku board vars
    board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    selected_cell: tuple = (0, 0)
    original_board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    
    # utility vars
    strategy : str = Solution.FAKE

    delay: float = 1

    start_time = 0
    end_time = 0

    count: int = 0

    def init_board(self):
        base_puzzle = [
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
        self.board = base_puzzle
        self.original_board = [row[:] for row in base_puzzle]
        self.count = 0
        self.start_time = 0
        self.end_time = 0

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

    def insert_cell(self, row: int, col: int, number: int):
        if self.original_board[row][col] == 0:
            self.board[row][col] = number
        
        self.selected_cell = (row, col)
        self.count += 1

    def delete_cell(self, row: int, col: int):
        if self.original_board[row][col] == 0:
            self.board[row][col] = 0

        self.selected_cell = (row, col)
        self.count += 1

# State를 통해 보드를 조작하고 분석할 수 있는 클래스입니다.
class SudokuSolver(State):
    async def solve(self):
        self.start_time = time.time()
        if self.strategy == Solution.FAKE:
            self.fake_solve()
        elif self.strategy == Solution.ConstraintPropagation:
            self.constraint_propatation_solve()
        elif self.strategy == Solution.BackTracking:
            self.backtracking()
        self.end_time = time.time()
    
    def fake_solve(self):
        for i in range(9):
            for j in range(9):
                self.insert_cell(i, j, 1)
                
    def constraint_propatation_solve(self):
        updated = False  # a flag to track if any updates
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    candidates = self.find_candidates(row, col) 
                    if len(candidates) == 1:
                        # If there is exactly one candidate, fill the cell
                        num = candidates.pop()
                        self.insert_cell(row, col, num)
                        updated = True
        if updated:
            self.constraint_propatation_solve()  # Continue until no more updates  
        else:
            self.backtracking()

    def find_candidates(self, row: int, col: int) -> set[int]:
        candidates = set(range(1, 10))
        # possible numbers that can be placed in the empty cell.
        candidates -= set(self.board[row])  # Remove numbers already present in the same row
        candidates -= {self.board[i][col] for i in range(9)}  # Remove numbers already present in the same column
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):  # Remove numbers already present in the 3x3 subgrid containing the cell
            for j in range(box_col, box_col + 3):
                candidates.discard(self.board[i][j])
        return candidates

    def backtracking(self):
        def backtrack(row: int, col: int) -> bool:
            # If we reach the end of the board, return True
            if row == 9:
                return True
            
            # If this cell is already filled, move to the next cell  
            if self.board[row][col] != 0:
                return backtrack(row + (col + 1) // 9, (col + 1) % 9)
            
            # Get the possible candidates for this cell
            candidates = self.find_candidates(row, col)
            
            for num in candidates:
                if self.is_valid_move(row, col, num):
                    self.insert_cell(row, col, num)
                    
                    # Recursively attempt to solve from the next cell
                    if backtrack(row + (col + 1) // 9, (col + 1) % 9):
                        return True
                    
                    # If no solution is found, backtrack and try next candidate
                    self.delete_cell(row, col)
            
            return False  # No valid solution found for this cell
    
    # Start backtracking from the top-left corner (0,0)
        return backtrack(0, 0)
 

# ----------------------------------------------------- Frontend Area --------------------------------------------------
def index():
    return rx.container(
        rx.flex(
            rx.hstack(
                rx.text(f'{State.count}회'),
                rx.text(f'{round((State.end_time - State.start_time)*10000) / 10000}초'),
                rx.button("생성", on_click = State.init_board),
                rx.select(
                    [Solution.FAKE, Solution.BackTracking, Solution.ConstraintPropagation],
                    value = State.strategy,
                    on_change = State.set_strategy
                ),
                rx.button("실행", on_click = SudokuSolver.solve)
            ),
            rx.vstack(
                rx.foreach(
                    State.board, 
                    lambda row: 
                        rx.hstack( 
                            rx.foreach(
                                row, 
                                lambda item: 
                                    rx.box(
                                        item,
                                        border="1px solid black",
                                        padding="0.5em",
                                        margin="0",
                                    ),
                            ),
                        )
                ),
                align_items="center",
            ),
            center_content=True,
            direction="column",
            justify="center",
            align="center",
            spacing="2",
        )
    )    


# ----------------------------------------------------- Page Area --------------------------------------------------
app = rx.App()
app.add_page(index)


