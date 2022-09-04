const HelloWorld = artifacts.require("HelloWorld");
const Owner = artifacts.require("Owner");
const EternalStorage = artifacts.require("EternalStorage");

module.exports = function(deployer) {
  deployer.deploy(Owner);
  deployer.deploy(EternalStorage);
};