package com.ips.kradziko.phd.core.training;

import org.javatuples.Pair;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

/**
 * Created by kradziko on 17/02/21.
 */

public abstract class ModelsTraining{
    private ModelsTraining models;

    ModelsTraining(){
        models = new JuliusModelsTraining("workdir_julius");
    }

    ModelsTraining(String workdir){
        models = new JuliusModelsTraining(workdir + "_julius");
    }

    ModelsTraining(int aligner, String workdir){
        if(aligner == 0)
            models = new JuliusModelsTraining(workdir + "_julius");
    }

    public abstract void train(String location);
}

class JuliusModelsTraining extends ModelsTraining{
    private String workdir;

    JuliusModelsTraining(String workdir){
        this.workdir = workdir;
    }

    public void train(String location){
        
    }

    private void generateGrammarFile(String fname, List<String> data){
        String path = workdir + "/grammar";
        new File(path).mkdirs();

        path += "/" + fname;
        try{
            PrintWriter writer = new PrintWriter(path, "UTF-8");
            data.stream().forEach(entry -> writer.println(String.format("S : NS_B %s NS_E", entry)));
            writer.close();
        } catch (IOException e) {
            // do something
        }
    }

    private void generateVocaFile(String fname, Map<String, List<Pair<String, String>>> data){
        String path = workdir + "/grammar";
        new File(path).mkdirs();

        path += "/" + fname;
        try{
            PrintWriter writer = new PrintWriter(path, "UTF-8");
            data.entrySet().stream().forEach(entry -> {
                writer.println(String.format("%% %s", entry.getKey()));
                entry.getValue().stream().forEach(
                        pair -> writer.println(String.format("%s\t%s", pair.getValue0(), pair.getValue1())));
                writer.println();
            });
            writer.close();
        } catch (IOException e) {
            // do something
        }
    }
}
