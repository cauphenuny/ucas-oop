package library;

public class Demo {
    public static void main(String[] args) {
        Librarian admin = new Librarian("A001", "管理员");
        admin.addBook(new Book("978-7-111-12345-6", "Java 编程", "张三"));
        admin.addBook(new Book("978-7-222-54321-0", "数据结构", "李四"));

        Student stu = new Student("202312345", "袁同学");

        admin.borrowBook(stu, "978-7-111-12345-6", 30);
        System.out.println("借书后：" + stu.getBorrowedBooks());

        admin.returnBook(stu, "978-7-111-12345-6");
        System.out.println("还书后：" + stu.getBorrowedBooks());
    }
}
