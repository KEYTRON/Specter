import sys
from PyQt6.QtWidgets import QApplication
from specter.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Specter")
    app.setOrganizationName("keytron46")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
