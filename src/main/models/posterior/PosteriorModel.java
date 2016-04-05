package main.models.posterior;

import main.models.AbstractModel;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by loongkuan on 1/04/16.
 **/
public abstract class PosteriorModel extends AbstractModel{
    public Instances dataSet;
    private ArrayList<ArrayList<String>> xPossibleValues;
    private Instances allPossibleInstances;

    public abstract void setData(Instances data);
    public abstract double findPyGv(double classValue, Instance vector);
    public abstract double findDistance(PosteriorModel model1, PosteriorModel model2, Instances domain);

    public Instances getAllPossibleInstances() {
        getAllPossibleInstances();
        return allPossibleInstances;
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
}
