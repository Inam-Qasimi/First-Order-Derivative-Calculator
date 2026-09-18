#include <iostream>
#include <cmath>
#include <vector>

// Explanation for program here: https://drive.google.com/file/d/1kBVdoFyESEyZhcLK2o_1udxIK9NVoRHq/view?usp=sharing
// Credit to Wikipedia for the beginning code and algebra rules for dual numbers



struct Dual
{
    double realPart, infinitesimalPart;

    Dual(double realPart, double infinitesimalPart = 0) : realPart(realPart), infinitesimalPart(infinitesimalPart) {}

    Dual operator-() const
    {
        return Dual (
            -this->realPart,
            -this->infinitesimalPart
        );
    }
};

// OPERATOR OVERLOADERS
Dual operator+(const Dual &LHS, const Dual &RHS)
{
    return Dual (
        LHS.realPart + RHS.realPart,
        LHS.infinitesimalPart + RHS.infinitesimalPart
    );
}

Dual operator-(const Dual &LHS, const Dual &RHS)
{
    return Dual (
        LHS.realPart - RHS.realPart,
        LHS.infinitesimalPart - RHS.infinitesimalPart
    );
}

Dual operator*(const Dual &LHS, const Dual &RHS)
{
    return Dual (
        LHS.realPart * RHS.realPart,
        LHS.infinitesimalPart * RHS.realPart + LHS.realPart * RHS.infinitesimalPart
    );
}

Dual operator/(const Dual &LHS, const Dual &RHS)
{
    return Dual (
        LHS.realPart / RHS.realPart,
        LHS.infinitesimalPart / RHS.realPart - (LHS.realPart * RHS.infinitesimalPart) / (RHS.realPart * RHS.realPart)
    );
}
// END OF OPERATOR OVERLOADING

// FUNCTION OVERLOADING
Dual sqrt(Dual const &other)
{
    double sqrtReal = std::sqrt(other.realPart);
    return Dual (
        sqrtReal,
        0.5 * (1 / sqrtReal) * other.infinitesimalPart
    );
}

Dual pow(Dual const &other, double const &base)
{
    return Dual (
        std::pow(other.realPart, base),
        base * std::pow(other.realPart, base - 1) * other.infinitesimalPart
    );
}

Dual log(Dual const &other)
{
    return Dual (
        std::log(other.realPart),
        1 / (other.realPart) * other.infinitesimalPart
    );
}

Dual sin(Dual const &other)
{
    return Dual (
        std::sin(other.realPart),
        other.infinitesimalPart * std::cos(other.realPart)
    );
}

Dual cos(Dual const &other)
{
    return Dual (
        std::cos(other.realPart),
        other.infinitesimalPart * (-1 * std::sin(other.realPart))
    );
}

Dual tan(Dual const &other)
{
    return Dual (
        std::tan(other.realPart),
        other.infinitesimalPart * 1 / (std::cos(other.realPart) * std::cos(other.realPart))
    );
}

Dual csc(Dual const &other)
{
    return Dual (
        1 / std::sin(other.realPart),
        other.infinitesimalPart * -1 * 1 / (std::sin(other.realPart) * std::tan(other.realPart))
    );
}

Dual sec(Dual const &other)
{
    return Dual (
        1 / std::cos(other.realPart),
        other.infinitesimalPart * (1 / std::cos(other.realPart)) * std::tan(other.realPart)
    );
}

Dual cot(Dual const &other)
{
    return Dual (
        1 / std::tan(other.realPart),
        other.infinitesimalPart * -1 * 1/ (std::sin(other.realPart) * std::sin(other.realPart))
    );
}
// END OF FUNCTION OVERLOADING

template <typename T>
T f(const T &x, const T &y, const T &z)
{
    return sin(x) * cos(y) * tan(z);
}

int main()
{
    Dual x = Dual(2); // Evaluate at x = 2
    Dual y = Dual(3); // Evaluate at y = 3
    Dual z = Dual(4); // Evaluate at z = 4 
    Dual epsilon = Dual(0,1);
    Dual dx = f(x + epsilon, y, z);
    Dual dy = f(x, y + epsilon, z);
    Dual dz = f(x, y, z + epsilon);

    std::cout << "dz/dx = " << dx.infinitesimalPart << ", " << "dz/dy = " << dy.infinitesimalPart << ", df/dz = " << dz.infinitesimalPart <<'\n';
    
    return 0;
}