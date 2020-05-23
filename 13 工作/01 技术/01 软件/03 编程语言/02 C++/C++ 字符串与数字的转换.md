# C++ 字符串与数字的转换

## 需要确认及补充的



# 缘由


今天在看到一个算法题：序列化与反序列化二叉树 的时候，题中总结了 字符串与数字之间的转换，因此这里也总结下：


# 整数转换为字符串




    #include <iostream>
    #include <sstream>
    #include <string>


    using namespace std;


    //itoa不是标准 C 语言函数所以不能在所有的编译器中使用(比如 gcc)
    //itoa并不是一个标准的 C 函数，它是 Windows 特有的，如果要写跨平台的程序，请用 sprintf。
    string IntToStringByitoa(int num) {
        char pstr[81];
        _itoa_s(num, pstr, 10);//10 是进制
        string str(pstr);
        cout << "int(" << num << ") to " << "str(" << str << ")" << endl;
        return str;
    }


    //  使用 sprintf 将任意的格式化信息输出到 char *中
    string IntToStringBysprintf(int num) {
        char pstr[81];
        sprintf_s(pstr, "%d", num);//%d 是十进制数，%x 是 16 进制
        string str(pstr);
        cout << "int(" << num << ") to " << "str(" << str << ")" << endl;
        return str;
    }


    //  使用 stringstream 进行字符串的格式化
    string IntToStringBystringstream(int num) {
        string str;
        stringstream ss;
        ss << num;
        ss >> str;
        cout << "int(" << num << ") to " << "str(" << str << ")" << endl;
        return str;
    }

    //  使用 ostringstream 将任意格式的信息转换为字符串(输出)
    string IntToStringByostringstream(int num) {
        string str;
        ostringstream ss;

        ss << num;
        str = ss.str();
        cout << "int(" << num << ") to " << "str(" << str << ")" << endl;
        return str;
    }

    int main() {
        IntToStringByitoa(25);
        IntToStringBysprintf(25);
        IntToStringBystringstream(25);
        IntToStringByostringstream(25);
        return EXIT_SUCCESS;
    }


输出：


    int(25) to str(25)
    int(25) to str(25)
    int(25) to str(25)
    int(25) to str(25)




# 字符串转换为整数




    #include <iostream>
    #include <vector>
    #include <sstream>
    #include <string>

    using namespace std;


    //  使用 atoi 将 char*转换为整数
    int StringToIntByatoi(string str) {
        int num = atoi(str.c_str());
        cout << "str(" << str << ") to " << "int(" << num << ")" << endl;
        return num;
    }

    //  使用 sscanf 将字符串转换为任意格式的信息
    int StringToIntBysscanf(string str) {
        int num;
        sscanf_s(str.c_str(), "%d", &num);
        cout << "str(" << str << ") to " << "int(" << num << ")" << endl;
        return num;
    }

    //  使用 stringstream 进行字符串的格式化(输入)
    int StringToIntBystringstream(string str) {
        int num;
        stringstream ss;
        ss << str;
        ss >> num;
        cout << "str(" << str << ") to " << "int(" << num << ")" << endl;
        return num;
    }

    //  使用 istringstream 将字符串转换为整数
    int StringToIntByistringstream(string str) {
        int num;
        istringstream ss;
        ss.str(str);
        ss >> num;
        cout << "str(" << str << ") to " << "int(" << num << ")" << endl;
        return num;
    }

    int main() {
        StringToIntByatoi("25");
        StringToIntBysscanf("25");
        StringToIntBystringstream("25");
        StringToIntByistringstream("25");
        return EXIT_SUCCESS;
    }


输出：


    str(25) to int(25)
    str(25) to int(25)
    str(25) to int(25)
    str(25) to int(25)







# 相关

1. [序列化二叉树](https://github.com/gatieme/CodingInterviews/tree/master/062-%E5%BA%8F%E5%88%97%E5%8C%96%E4%BA%8C%E5%8F%89%E6%A0%91)
