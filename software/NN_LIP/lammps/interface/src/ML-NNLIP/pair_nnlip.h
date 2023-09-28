#ifdef PAIR_CLASS

PairStyle(nnlip, PairNNLIP) // nnlip is the name in the input script

#else

#ifndef LMP_PAIR_NNLIP_H
#define LMP_PAIR_NNLIP_H

#include "pair.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

namespace LAMMPS_NS
{
    class PairNNLIP : public Pair
    {
    public:
        torch::jit::script::Module module;
        PairNNLIP(class LAMMPS *);
        virtual ~PairNNLIP();
        virtual void compute(int, int);
        virtual void init_style();
        virtual double init_one(int, int);
        virtual void settings(int, char **);
        virtual void coeff(int, char **);
        virtual int select_gpu();

    protected:
        virtual void allocate();
        double cutoff;
        double cutoffsq;
        int system_type = 0;
        std::string datatype;
        torch::Dtype tensor_type = torch::kDouble;
        torch::TensorOptions option1 = torch::TensorOptions().dtype(torch::kDouble);
        torch::TensorOptions option2 = torch::TensorOptions().dtype(torch::kLong);
        torch::Tensor device_tensor = torch::empty(1);
    };
}

#endif
#endif
