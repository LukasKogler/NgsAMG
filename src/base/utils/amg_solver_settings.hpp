#ifndef FILE_AMG_SOLVER_SETTINGS_HPP
#define FILE_AMG_SOLVER_SETTINGS_HPP

#include <memory>
#include <vector>

namespace ngcore
{
  class Flags;
}

namespace ngla
{
  class BaseMatrix;
}


namespace amg
{

class AMGSolverSettings
{
public:
  AMGSolverSettings()  = default;
  ~AMGSolverSettings() = default;

  virtual void set(std::string const &key, int         const &val) = 0;
  virtual void set(std::string const &key, double      const &val) = 0;
  virtual void setBool(std::string const &key, bool        const &val) = 0;
  virtual void set(std::string const &key, std::string const &val) = 0;

  virtual void set(std::string const &key, std::vector<std::string> const &vals) = 0;
  virtual void set(std::string const &key, std::vector<int> const &vals) = 0;
  virtual void set(std::string const &key, std::vector<double> const &vals) = 0;
  virtual void set(std::string const &key, std::vector<float> const &vals) = 0;
  virtual void setBool(std::string const &key, std::vector<bool> const &vals) = 0;

  virtual void setDirichletList(std::vector<int> &dirichletRows) = 0;
  virtual std::vector<int> &getDirichletList() = 0;

  // for elasticity AMG
  virtual void setVertexCoordinates(std::vector<double> &vertexCoordinates) = 0;
  virtual std::vector<double> &getVertexCoordinates() = 0;
  virtual std::vector<double> const &getVertexCoordinates() const = 0;

  // provide nodal-p2 mapping
  struct NodalP2Triple
  {
    unsigned vI;
    unsigned vJ;
    unsigned vMid;
  };

  virtual void setNodalP2Connectivity(std::vector<NodalP2Triple> &p2Triples) = 0;
  virtual std::vector<NodalP2Triple>       &getNodalP2Connectivity()       = 0;
  virtual std::vector<NodalP2Triple> const &getNodalP2Connectivity() const = 0;
  
  virtual ngcore::Flags const& getFlags() const = 0;

  virtual bool checkValidity() const = 0;

  virtual std::string getSolverType() const = 0;
  virtual std::string getAMGType() const = 0;

}; // class AMGSolverSettings  

} // namespace amg  



#endif  // FILE_AMG_SOLVER_SETTINGS_HPP