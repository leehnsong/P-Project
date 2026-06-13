import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

// 네이버 지도 클라이언트 ID는 소스에 하드코딩하지 않는다.
// 우선순위: local.properties(권장, Android Studio용) → gradle property → 환경변수(터미널/run.sh용)
val naverMapClientId: String = run {
    val props = Properties()
    val localFile = rootProject.file("local.properties")
    if (localFile.exists()) {
        localFile.inputStream().use { props.load(it) }
    }
    props.getProperty("NAVER_MAP_CLIENT_ID")
        ?: providers.gradleProperty("NAVER_MAP_CLIENT_ID").orNull
        ?: System.getenv("SMARTPARKING_NAVER_MAP_CLIENT_ID")
        ?: ""
}

android {
    namespace = "com.example.pproject"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.pproject"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        manifestPlaceholders["naverMapClientId"] = naverMapClientId

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    implementation("androidx.cardview:cardview:1.0.0")
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

    //retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.retrofit2:converter-scalars:2.9.0")
    implementation("com.naver.maps:map-sdk:3.23.2")
}
