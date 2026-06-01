import fitz # PyMuPDF 패키지
import os

def export_pdf_to_png(pdf_path, output_folder):
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f" -> 출력 폴더 '{output_folder}'를 생성하였습니다.")
    
    # PDF 문서 열기
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f" -> 총 {total_pages}개의 슬라이드 페이지를 로드하였습니다.")
    
    for i, page in enumerate(doc):
        # 고해상도 선명함을 위해 2.0배 줌 매트릭스 설정 (DPI=200 상당)
        # 구글 프레젠테이션 배경 화면으로 적용 시 글자가 아주 선명하게 보입니다.
        zoom = 2.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        # 파일명 정의 (slide_01.png, slide_02.png, ...)
        file_name = f"slide_{i+1:02d}.png"
        save_path = os.path.join(output_folder, file_name)
        
        # PNG 저장
        pix.save(save_path)
        print(f"  [성공] {file_name} 저장 완료")
        
    print(f"\n[완료] 총 {total_pages}개의 모든 슬라이드가 고해상도 PNG 이미지로 일괄 변환되었습니다.")

if __name__ == "__main__":
    pdf_file = "presentation.pdf"
    output_dir = "slides"
    
    if os.path.exists(pdf_file):
        export_pdf_to_png(pdf_file, output_dir)
    else:
        print(f"[에러] {pdf_file} 파일이 존재하지 않습니다. 먼저 LaTeX 컴파일을 완료해야 합니다.")
