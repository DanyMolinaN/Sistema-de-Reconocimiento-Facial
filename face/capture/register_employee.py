import cv2
import os
import uuid
from datetime import datetime

SAVE_DIR = "data/employees/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def register_employee(employee_id=None):
    if employee_id is None:
        employee_id = str(uuid.uuid4())

    cap = cv2.VideoCapture(0)
    print("[INFO] Presiona 's' para capturar | 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Mire al frente - sin gafas ni mascarilla",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Registro de empleado", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{employee_id}_{timestamp}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"[OK] Imagen guardada: {path}")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return employee_id


if __name__ == "__main__":
    emp_id = input("Ingrese ID del empleado (Enter para auto): ").strip()
    emp_id = emp_id if emp_id else None
    register_employee(emp_id)
