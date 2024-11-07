from typing import List, Optional
import reflex as rx
import asyncio
import time
from enum import Enum

# ----------------------------------------------------- Backend Area --------------------------------------------------
class Solution:
    FAKE = "Fake"
    ConstraintPropagation = "Constratint Propatation"
        
class SudokuState(rx.State):
    # 9x9 sudoku board class variable that all instances share
    board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    selected_cell: tuple = (0, 0)
    original_board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    strategy : str = Solution.FAKE
    delay: float = 1

    def init_board(cls):
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
        cls.board = base_puzzle
        cls.original_board = [row[:] for row in base_puzzle]

    def is_valid_move(cls, row: int, col: int, number: int) -> bool:
        # check row
        if number in cls.board[row]:
            return False

        # check column
        if number in [cls.board[i][col] for i in range(9)]:
            return False

        # check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if cls.board[i][j] == number:
                    return False

        return True

    def insert_cell(cls, row: int, col: int, number: int):
        if cls.original_board[row][col] == 0:
            cls.board[row][col] = number

        cls.selected_cell = (row, col)
        cls.common_update()

    def delete_cell(cls, row: int, col: int):
        if cls.original_board[row][col] == 0:
            cls.board[row][col] = 0

        cls.selected_cell = (row, col)
        cls.common_update()

    def common_update(cls):
        pass

    def set_strategy(cls, strategy):
        cls.strategy = strategy

class FakeSudokuSolver(SudokuState):
    def solve(self):
        for i in range(9):
            for j in range(9):
                self.insert_cell(i, j, 1)

class ConstraintPropagationSudokuSolver(SudokuState):
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

    def solve(self):
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
            self.solve()  # Continue until no more updates  
        else:
            self.backtracking()

# ----------------------------------------------------- Frontend Area --------------------------------------------------
def display_row(row):
    return rx.hstack(
        rx.foreach(
            row,
            lambda item: rx.box(
                item,
                border="1px solid black",
                padding="0.5em",
                margin="0",
            ),
        ),
    )

def button():
    return rx.button(

    )
def index():
    return rx.container(
        rx.flex(
            rx.hstack(
                rx.button("생성", on_click = SudokuState.init_board),
                rx.select(
                    [Solution.FAKE, Solution.ConstraintPropagation],
                    value = SudokuState.strategy,
                    on_change = SudokuState.set_strategy
                ),
                rx.cond(
                    (SudokuState.strategy == Solution.FAKE), # 조건
                    rx.button("실행", on_click = FakeSudokuSolver.solve), # 참 
                    rx.cond( # 거짓
                        (SudokuState.strategy == Solution.ConstraintPropagation), # 조건
                        rx.button("실행", on_click = ConstraintPropagationSudokuSolver.solve), # 참
                        rx.button("실행") # 거짓
                    )
                )
            ),
            rx.vstack(
                rx.foreach(SudokuState.board, display_row),
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


