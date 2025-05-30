import shutil
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
import logging

class MenuAwareRichHandler(RichHandler):
    def __init__(self, menu_text="", terminal_width=None):
        super().__init__(rich_tracebacks=True, markup=True)
        self.menu_text = menu_text
        self.terminal_width = terminal_width or shutil.get_terminal_size().columns
        self.log_buffer = []
        self.max_buffer_lines = 20  # Maximum number of log lines to keep above menu
        self.console = Console()

    def emit(self, record):
        try:
            # Format the message using Rich
            message = self.format(record)
            
            # Add styling based on log level
            style = self._get_style_for_level(record.levelno)
            styled_message = Text(message, style=style)
            
            # Add to buffer
            self.log_buffer.append(styled_message)
            if len(self.log_buffer) > self.max_buffer_lines:
                self.log_buffer.pop(0)
            
            self._redraw_screen()
        except Exception:
            self.handleError(record)

    def _get_style_for_level(self, levelno):
        if levelno >= logging.ERROR:
            return "red"
        elif levelno >= logging.WARNING:
            return "yellow"
        elif levelno >= logging.INFO:
            return "blue"
        return "white"

    def _redraw_screen(self):
        # Clear screen and move cursor to top
        self.console.clear()
        
        # Print log buffer
        for message in self.log_buffer:
            self.console.print(message)
        
        # Print separator
        self.console.print("─" * self.terminal_width, style="dim")
        
        # Print menu
        self.console.print(self.menu_text, style="bold green")

    def update_menu(self, menu_text):
        self.menu_text = menu_text
        self._redraw_screen()