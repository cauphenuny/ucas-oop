package library;

import java.time.LocalDate;
import java.util.Objects;

/**
 * 书籍实体：描述一本书的基本信息及借阅状态。
 */
public class Book {
    private final String isbn;           // 国际标准书号，作为唯一标识
    private String title;                // 书名
    private String author;               // 作者
    private boolean borrowed;            // 是否已被借出
    private String borrowedByStudentId;  // 借阅者学号
    private LocalDate borrowedDate;      // 借出日期
    private LocalDate dueDate;           // 应还日期

    public Book(String isbn, String title, String author) {
        this.isbn = Objects.requireNonNull(isbn, "isbn不能为空");
        this.title = Objects.requireNonNull(title, "title不能为空");
        this.author = Objects.requireNonNull(author, "author不能为空");
    }

    void markBorrowed(String studentId, LocalDate borrowedDate, LocalDate dueDate) {
        if (borrowed) {
            throw new IllegalStateException("书籍已被借出");
        }
        this.borrowed = true;
        this.borrowedByStudentId = studentId;
        this.borrowedDate = borrowedDate;
        this.dueDate = dueDate;
    }

    void markReturned() {
        if (!borrowed) {
            throw new IllegalStateException("书籍未被借出");
        }
        this.borrowed = false;
        this.borrowedByStudentId = null;
        this.borrowedDate = null;
        this.dueDate = null;
    }

    public boolean isOverDue(LocalDate today) {
        return borrowed && dueDate != null && today.isAfter(dueDate);
    }

    public String getIsbn() { return isbn; }
    public String getTitle() { return title; }
    public String getAuthor() { return author; }
    public boolean isBorrowed() { return borrowed; }
    public String getBorrowedByStudentId() { return borrowedByStudentId; }
    public LocalDate getBorrowedDate() { return borrowedDate; }
    public LocalDate getDueDate() { return dueDate; }

    public void setTitle(String title) { this.title = Objects.requireNonNull(title); }
    public void setAuthor(String author) { this.author = Objects.requireNonNull(author); }

    @Override
    public String toString() {
        return "Book{"
            + "isbn='" + isbn + '\'' + ", title='" + title + '\'' + ", author='" + author + '\'' +
            ", borrowed=" + borrowed +
            (borrowed ? ", borrowedBy=" + borrowedByStudentId + ", due=" + dueDate : "") + '}';
    }
}
