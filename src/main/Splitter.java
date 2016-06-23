package main;

import moa.recommender.rc.utils.Hash;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by Lee on 23/06/2016.
 */
public class Splitter {
    public static void main(String[] args) {
        /*
        String[] attributes = new String{"band1", "band2", "band3", "band4", "band5", "band6", "band7",
                "NDVI", "NDWI", "brightness"};
                */
        try{
            Instances data = loadDataSet("./datasets/train_seed0.arff");
            System.out.println("Parsing data...");
            // Get id attribute
            String idAttributeName = data.attribute(0).name();
            // Get all non-class attributes
            HashSet<String> attributeSet = new HashSet<>();
            HashSet<String> dateSet = new HashSet<>();
            for (int i = 1; i < data.numAttributes() - 1; i++) {
                String[] year_att = data.attribute(i).name().split("_");
                dateSet.add(year_att[0]);
                attributeSet.add(year_att[1]);
            }
            String[] attributes = attributeSet.toArray(new String[attributeSet.size()]);
            String [] dates = dateSet.toArray(new String[dateSet.size()]);
            // Get class attribute and values
            String className = data.attribute(data.numAttributes() - 1).name();
            ArrayList<String> classValues = new ArrayList<>();
            for (int i = 0; i < data.attribute(data.numAttributes() - 1).numValues(); i++) {
                classValues.add(data.attribute(data.numAttributes() - 1).value(i));
            }

            System.out.println("Creating separate data...");
            // Create new Instances
            Instances[] splitDataSets = new Instances[dates.length];
            for (int i = 0; i < dates.length; i++) {
                ArrayList<Attribute> dataAttributes = new ArrayList<>();
                // Add idAttribute
                dataAttributes.add(new Attribute(idAttributeName));
                // Add feature Attributes
                for (int j = 0; j < attributes.length; j++) {
                    // Attribute is set to numerical by default
                    dataAttributes.add(new Attribute(attributes[j]));
                }
                // Add Class Attribute
                dataAttributes.add(new Attribute(className, classValues));
                splitDataSets[i] = new Instances(dates[i], dataAttributes, data.size());
                splitDataSets[i].setClassIndex(splitDataSets[i].numAttributes() - 1);
                splitDataSets[i].setRelationName(dates[i]);
            }

            // Add relevant data to created Instances
            System.out.println("Splitting data...");
            while (data.size() > 0) {
                Instance instance = data.instance(0);
                double[][] attributesValues = new double[dates.length][2 + attributes.length];
                // Get feature attributes
                for (int i = 1; i < instance.numAttributes() - 1; i++) {
                    String[] dates_att = instance.attribute(i).name().split("_");
                    int dateIndex = ArrayUtils.indexOf(dates, dates_att[0]);
                    int attIndex = ArrayUtils.indexOf(attributes, dates_att[1]);
                    attributesValues[dateIndex][attIndex] = instance.value(i);
                }
                // Get id and class attribute and add each array as an instance to separate data sets
                for (int i = 0; i < dates.length; i++) {
                    attributesValues[i][0] = instance.value(0);
                    attributesValues[i][attributesValues[i].length - 1] = instance.value(instance.numAttributes() - 1);
                    splitDataSets[i].add(new DenseInstance(1.0, attributesValues[i]));
                }
                data.delete(0);
                System.gc();
            }

            // Write splitDataSets to separate files
            for (int i = 0; i < splitDataSets.length; i++) {
                System.out.println("Creating " + splitDataSets[i].relationName() + " file...");
                ArffSaver saver = new ArffSaver();
                saver.setInstances(splitDataSets[i]);
                saver.setFile(new File("./datasets/" + splitDataSets[i].relationName()));
                saver.setDestination(new File("./datasets/" + splitDataSets[i].relationName()));
                saver.writeBatch();
            }
        }
        catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static Instances loadDataSet(String filename) throws IOException {
        System.out.println("Reading data file...");
        // Check if any attribute is numeric
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        Instances result = new Instances(reader);
        result.setClassIndex(result.numAttributes() - 1);
        reader.close();
        return result;
    }

}
