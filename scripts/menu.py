#!/usr/bin/env python3
"""Interactive menu for Karina Voice Notification Generator."""

import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.table import Table
from rich import box
import readchar


console = Console()


class InteractiveMenu:
    """Beautiful interactive menu with arrow key navigation."""

    def __init__(self, title: str, options: list[dict], subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.options = options
        self.selected = 0
        self.console = Console()

    def _render(self) -> Panel:
        """Render the menu."""
        # Create menu items
        menu_text = Text()
        
        for i, opt in enumerate(self.options):
            if i == self.selected:
                # Selected item - highlighted
                menu_text.append("  ▸ ", style="bold cyan")
                menu_text.append(f"{opt['label']}\n", style="bold white on blue")
                if opt.get('desc'):
                    menu_text.append(f"    {opt['desc']}\n", style="dim cyan")
            else:
                menu_text.append("    ", style="dim")
                menu_text.append(f"{opt['label']}\n", style="white")
                if opt.get('desc'):
                    menu_text.append(f"    {opt['desc']}\n", style="dim")
            
            if i < len(self.options) - 1:
                menu_text.append("\n")

        # Footer
        footer = Text("\n  ↑↓ 이동  •  Enter 선택  •  q 종료", style="dim")
        menu_text.append(footer)

        # Create panel
        panel = Panel(
            Align.left(menu_text),
            title=f"[bold magenta]✨ {self.title}[/bold magenta]",
            subtitle=f"[dim]{self.subtitle}[/dim]" if self.subtitle else None,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        
        return panel

    def run(self) -> int | None:
        """Run the interactive menu. Returns selected index or None if cancelled."""
        with Live(self._render(), console=self.console, refresh_per_second=30, transient=True) as live:
            while True:
                key = readchar.readkey()
                
                if key == readchar.key.UP:
                    self.selected = (self.selected - 1) % len(self.options)
                elif key == readchar.key.DOWN:
                    self.selected = (self.selected + 1) % len(self.options)
                elif key == readchar.key.ENTER:
                    return self.selected
                elif key.lower() == 'q' or key == readchar.key.ESCAPE:
                    return None
                
                live.update(self._render())


def show_main_menu() -> str | None:
    """Show main menu and return selected action."""
    options = [
        {
            "label": "🎬 전체 파이프라인 실행",
            "desc": "다운로드 → 분할 → 전사 → TTS 생성",
            "action": "full"
        },
        {
            "label": "📥 음성 다운로드 & 추출",
            "desc": "YouTube에서 음성 다운로드 후 세그먼트 분할",
            "action": "download"
        },
        {
            "label": "📝 전사(Transcribe)부터 시작",
            "desc": "기존 오디오로 전사 → TTS 생성",
            "action": "transcribe"
        },
        {
            "label": "🎤 음성 생성만",
            "desc": "기존 전사 결과로 TTS 음성만 생성",
            "action": "generate"
        },
        {
            "label": "❌ 종료",
            "desc": "",
            "action": "exit"
        },
    ]

    menu = InteractiveMenu(
        title="Karina Voice Generator",
        subtitle="aespa 카리나 음성으로 Claude Code 알림음 생성",
        options=options
    )

    result = menu.run()
    
    if result is None:
        return None
    
    return options[result]["action"]


def show_segment_menu(segments: list) -> int | None:
    """Show segment selection menu."""
    options = [
        {"label": f"🎵 {seg.name}", "desc": ""} 
        for seg in segments
    ]
    options.append({"label": "❌ 취소", "desc": ""})

    menu = InteractiveMenu(
        title="세그먼트 선택",
        subtitle="깨끗한 음성 구간을 선택하세요",
        options=options
    )

    result = menu.run()
    
    if result is None or result == len(segments):
        return None
    
    return result


if __name__ == "__main__":
    # Test menu
    action = show_main_menu()
    if action:
        console.print(f"\n선택: [bold cyan]{action}[/bold cyan]")
    else:
        console.print("\n[dim]취소됨[/dim]")
