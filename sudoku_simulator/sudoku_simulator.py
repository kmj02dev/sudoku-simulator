from typing import List, Optional
import reflex as rx
import asyncio
import time

# ----------------------------------------------------- Backend Area --------------------------------------------------
        
class SudokuState(rx.State):
    # 9x9 sudoku board class variable that all instances share
    board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    selected_cell: tuple = (0, 0)
    original_board: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]
    strategy: str = "fake"
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

    def solve(cls):
        if cls.strategy == "fake":
            cls.fake_solution()
    
    def set_strategy(cls, strategy):
        cls.strategy = strategy

    @rx.background
    async def fake_solution(cls):
        for i in range(9):
            for j in range(9):
                yield SudokuState.insert_cell(i, j, 2)
                await asyncio.sleep(0.1)


    # 아래에 알고리즘을 추가
    # 알고리즘에서 보드를 조작할 때 insert_cell, delete_cell 함수를 무조건 사용하여야 함

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


def index():
    return rx.container(
        rx.flex(
            rx.hstack(
                rx.button("생성", on_click = SudokuState.init_board),
                rx.button("실행", on_click = SudokuState.fake_solution) 
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
