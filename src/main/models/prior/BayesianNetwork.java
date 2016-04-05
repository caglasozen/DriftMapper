package main.models.prior;

import explorer.ChordalysisModelling;
import main.generator.componets.BayesianNetworkGenerator;
import main.models.distance.Distance;
import main.models.distance.HellingerDistance;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 1/04/16.
 **/
public class BayesianNetwork extends PriorModel{
    private BayesianNetworkGenerator bnModel;

    public BayesianNetwork(Instances data) {
        this.dataSet = data;
        this.reset();
    }

    private BayesianNetwork(Instances data, BayesianNetworkGenerator bn) {
        this.dataSet = data;
        getAllPossibleValues();
        getAllPossibleInstances();
        this.bnModel = bn;
    }

    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        this.reset();
    }

    private void generateBayesNet() {
        String[] variablesNames = new String[dataSet.numAttributes()];
        for (int i = 0; i < variablesNames.length; i++) {
            variablesNames[i] = dataSet.attribute(i).name();
        }

        Instances trimmedInstances = trimClass(dataSet);
        // Chordalysis modeler with 0.1G of memory allocated
        ChordalysisModelling modeller = new ChordalysisModelling(0.05);
        modeller.buildModel(new Instances(trimmedInstances));

        bnModel = new BayesianNetworkGenerator(modeller, variablesNames, xPossibleValues);
    }

    private void getAllPossibleValues() {
        xPossibleValues = new ArrayList<>();

        // NOTE: j represents x_n
        for (int j = 0; j < dataSet.numAttributes() - 1; j++) {
            // Initialise local variables for this loop (aka. data for this X variable)
            HashSet<String> possibleValues = new HashSet<>();
            for (int i = 0; i < dataSet.size(); i++) {
                String curKey = Double.toString(dataSet.instance(i).value(j));
                if (!possibleValues.contains(curKey)) possibleValues.add(curKey);
            }
            // Add all the keys/Possible values to xPossibleValues
            xPossibleValues.add(new ArrayList<>(possibleValues));
        }
        //Suggest Garbage collector to run
        System.gc();
    }

    private void getAllPossibleInstances() {
        // Calculate and allocate space needed to store all possible instances
        int nCombinations = 1;
        for (ArrayList<String> values : xPossibleValues) {
            nCombinations *= values.size();
        }
        allPossibleInstances = new Instances(dataSet, nCombinations);

        // Generate set of all appeared combinations of x values
        generateCombinations(0, new ArrayList<>());

        //Suggest Garbage collector to run
        System.gc();
    }

    private void generateCombinations(int currentIndex, ArrayList<Double> auxCombination){
        if (currentIndex == this.xPossibleValues.size()){
            Instance inst = new DenseInstance(dataSet.numAttributes());
            inst.setDataset(dataSet);
            for (int j = 0; j < auxCombination.size(); j++) {
                inst.setValue(j, auxCombination.get(j));
            }
            allPossibleInstances.add(inst);
        }
        else {
            ArrayList<String> curValues = this.xPossibleValues.get(currentIndex);
            for (String value : curValues){
                // Add the new possible x value to the list of values and call itself with the new list
                auxCombination.add(Double.parseDouble(value));
                // Add the generated combinations with the new list to the existing combinations
                generateCombinations(currentIndex + 1, auxCombination);
                // Remove the added x value from the list of values
                auxCombination.remove(auxCombination.size() - 1);
            }
        }
    }

    @Override
    public BayesianNetwork copy() {
        return new BayesianNetwork(this.dataSet, new BayesianNetworkGenerator(this.bnModel));
    }

    @Override
    public void reset() {
        getAllPossibleValues();
        getAllPossibleInstances();
        generateBayesNet();
    }

    @Override
    public double findPv(double[] vector) {
        try {
            int[] dataVector = new int[vector.length];
            for (int i = 0; i < vector.length; i++) {
                dataVector[i] = this.bnModel.nodes[i].getOutcomeIndex(Double.toString(vector[i]));
            }
            return bnModel.getJointProbabilityOfX(dataVector);
        }
        catch (IllegalArgumentException ex) {
            System.out.print("e ");
            return 0.0f;
        }
    }

    @Override
    public double findDistance(PriorModel model1, PriorModel model2, Instances domain) {
        // Trim last attribute as allPossibleCombinations contains a class attribute which has NaN
        domain = trimClass(domain);
        Distance distanceMetric = new HellingerDistance();

        double[] p = new double[domain.size()];
        double[] q = new double[domain.size()];
        for (int i = 0; i < domain.size(); i++) {
            Instance inst = domain.get(i);
            p[i] = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, domain.numAttributes()));
            q[i] = model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, domain.numAttributes()));
        }
        return distanceMetric.findDistance(p, q);
    }
}
