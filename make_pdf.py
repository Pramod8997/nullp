from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 24)
        self.set_text_color(0, 102, 204)
        self.cell(0, 20, "The Magic Smart House", 0, 1, "C")
        self.ln(10)

    def chapter_title(self, num, title):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(204, 0, 0)
        self.cell(0, 10, f"Friend #{num}: {title}", 0, 1, "L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 14)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, body)
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", "", 16)
pdf.multi_cell(0, 10, "Imagine you have a magic toy house! Inside this house, you have four special friends who help you:")
pdf.ln(10)

pdf.chapter_title(1, "The Big Bodyguard")
pdf.chapter_body("This is a superhero who watches the house. If the house gets too hot or uses too much magic power, the superhero turns things off to keep you safe! He never sleeps.")

pdf.chapter_title(2, "The Smart Brain")
pdf.chapter_body("This brain looks at all your toys. It can say, 'Oh, that is the TV!' or 'I don't know what this toy is!'. It is like a fun guessing game.")

pdf.chapter_title(3, "The Helper Elf")
pdf.chapter_body("This little elf turns the toys on and off to save your shiny coins, but the elf always makes sure you are warm and cozy. The elf learns what makes you happy!")

pdf.chapter_title(4, "The Magic Screen")
pdf.chapter_body("A shiny picture frame where you can see all your toys playing and see the Helper Elf working!")

pdf.output("Magic_House.pdf")
